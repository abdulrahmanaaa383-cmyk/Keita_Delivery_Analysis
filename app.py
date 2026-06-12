import streamlit as st
import anthropic
import base64
import json
import pandas as pd
from io import BytesIO
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rider Performance Tracker",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0f1117; }

  .header-block {
    background: linear-gradient(135deg, #1a1d27 0%, #12151f 100%);
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
  }
  .header-block h1 { color: #ffffff; font-size: 28px; font-weight: 700; margin: 0; }
  .header-block p  { color: #8b8fa8; font-size: 14px; margin: 6px 0 0; }

  .metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }
  .metric-label { color: #8b8fa8; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }
  .metric-value { color: #ffffff; font-size: 28px; font-weight: 700; margin-top: 4px; }
  .metric-sub   { color: #5a9e6f; font-size: 12px; margin-top: 2px; }

  .rider-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .rider-name  { color: #ffffff; font-size: 15px; font-weight: 600; }
  .rider-id    { color: #8b8fa8; font-size: 12px; margin-top: 2px; }

  .badge-working { background: #1a3a2a; color: #4ade80; border: 1px solid #166534;
                   border-radius: 6px; padding: 2px 10px; font-size: 12px; font-weight: 600; }
  .badge-offline { background: #2a1a1a; color: #f87171; border: 1px solid #7f1d1d;
                   border-radius: 6px; padding: 2px 10px; font-size: 12px; font-weight: 600; }
  .badge-idle    { background: #2a2515; color: #fbbf24; border: 1px solid #78350f;
                   border-radius: 6px; padding: 2px 10px; font-size: 12px; font-weight: 600; }

  .upload-zone {
    background: #1a1d27;
    border: 2px dashed #2a2d3a;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    margin-bottom: 20px;
  }

  div[data-testid="stFileUploader"] {
    background: #1a1d27;
    border: 2px dashed #3a3d50;
    border-radius: 12px;
    padding: 20px;
  }

  div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
    width: 100%;
    cursor: pointer;
    transition: opacity .2s;
  }
  div.stButton > button:hover { opacity: .88; }

  .stDataFrame { border-radius: 10px; overflow: hidden; }

  .section-title {
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2d3a;
  }

  .info-msg {
    background: #1a2535;
    border: 1px solid #1d4ed8;
    border-radius: 8px;
    padding: 12px 16px;
    color: #93c5fd;
    font-size: 13px;
    margin-bottom: 12px;
  }
  .success-msg {
    background: #162520;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 12px 16px;
    color: #4ade80;
    font-size: 13px;
    margin-bottom: 12px;
  }
  .error-msg {
    background: #251616;
    border: 1px solid #7f1d1d;
    border-radius: 8px;
    padding: 12px 16px;
    color: #f87171;
    font-size: 13px;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "all_riders" not in st.session_state:
    st.session_state.all_riders = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(uploaded_file) -> str:
    return base64.standard_b64encode(uploaded_file.read()).decode("utf-8")


def extract_riders_from_image(client: anthropic.Anthropic, img_b64: str, media_type: str) -> list[dict]:
    """Send image to Claude and get structured rider data."""
    prompt = """
You are a data extraction assistant. Analyze this rider/delivery dashboard screenshot carefully.

Extract ALL riders visible. For EACH rider return a JSON object with these exact keys:
- rider_id: the numeric ID (string)
- name: rider's display name
- session: current session time range (e.g. "18:00 - 00:00")
- utr: UTR value as float (e.g. 0.6)
- deliveries: number shown next to "Deliveries" as integer  
- accepted: number of accepted orders (from the donut chart label or "Accepted" field) as integer
- stacked: stacked orders count as integer (0 if not shown)
- acceptance_rate: percentage as float (e.g. 50.0)
- cash_balance: numeric value only, no currency symbol (e.g. 197.36)
- currency: currency code (SAR, EGP, USD, etc.)
- rider_state: "Working", "Offline", or "Idle"
- location_area: location/area text if visible, else ""

Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
Example: [{"rider_id":"111931","name":"Mohammad Tanbir","session":"18:00-00:00","utr":0.6,"deliveries":1,"accepted":9,"stacked":2,"acceptance_rate":50.0,"cash_balance":197.36,"currency":"SAR","rider_state":"Working","location_area":"Riyadh South"}]

If no riders found, return [].
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    raw = response.content[0].text.strip()
    # clean fences just in case
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def build_excel(riders: list[dict]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Riders Report"

    # ── Styles ────────────────────────────────────────────────────────────────
    header_font   = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    header_fill   = PatternFill("solid", start_color="4F46E5")
    center        = Alignment(horizontal="center", vertical="center")
    left          = Alignment(horizontal="left",   vertical="center")
    thin          = Side(style="thin", color="D1D5DB")
    border        = Border(left=thin, right=thin, top=thin, bottom=thin)

    working_fill  = PatternFill("solid", start_color="D1FAE5")
    offline_fill  = PatternFill("solid", start_color="FEE2E2")
    idle_fill     = PatternFill("solid", start_color="FEF3C7")

    alt_fill      = PatternFill("solid", start_color="F9FAFB")

    # ── Headers ───────────────────────────────────────────────────────────────
    headers = [
        "Rider ID", "Name", "Session", "UTR", "Deliveries (System)",
        "Accepted (Real)", "Stacked", "Acceptance Rate %",
        "Cash Balance", "Currency", "Rider State", "Location"
    ]
    col_widths = [12, 28, 16, 8, 18, 16, 10, 20, 14, 10, 14, 24]

    for col, (h, w) in enumerate(zip(headers, col_widths), 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font      = header_font
        cell.fill      = header_fill
        cell.alignment = center
        cell.border    = border
        ws.column_dimensions[get_column_letter(col)].width = w

    ws.row_dimensions[1].height = 28

    # ── Data rows ─────────────────────────────────────────────────────────────
    for r_idx, rider in enumerate(riders, 2):
        row = [
            rider.get("rider_id", ""),
            rider.get("name", ""),
            rider.get("session", ""),
            rider.get("utr", ""),
            rider.get("deliveries", ""),
            rider.get("accepted", ""),
            rider.get("stacked", ""),
            rider.get("acceptance_rate", ""),
            rider.get("cash_balance", ""),
            rider.get("currency", ""),
            rider.get("rider_state", ""),
            rider.get("location_area", ""),
        ]
        state = str(rider.get("rider_state", "")).lower()
        row_fill = (working_fill if "work" in state
                    else offline_fill if "off" in state
                    else idle_fill if "idle" in state
                    else (alt_fill if r_idx % 2 == 0 else None))

        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.border = border
            cell.alignment = center if c_idx != 2 else left
            if row_fill:
                cell.fill = row_fill

        ws.row_dimensions[r_idx].height = 22

    # ── Summary sheet ─────────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Summary")
    ws2["A1"] = "Report Generated"
    ws2["B1"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    ws2["A2"] = "Total Riders"
    ws2["B2"] = len(riders)
    ws2["A3"] = "Working"
    ws2["B3"] = sum(1 for r in riders if "work" in str(r.get("rider_state","")).lower())
    ws2["A4"] = "Offline"
    ws2["B4"] = sum(1 for r in riders if "off"  in str(r.get("rider_state","")).lower())
    ws2["A5"] = "Avg UTR"
    utrs = [r["utr"] for r in riders if isinstance(r.get("utr"), (int, float))]
    ws2["B5"] = round(sum(utrs)/len(utrs), 2) if utrs else 0
    ws2["A6"] = "Total Accepted Orders"
    ws2["B6"] = sum(r.get("accepted", 0) or 0 for r in riders)

    for cell in ["A1","A2","A3","A4","A5","A6"]:
        ws2[cell].font = Font(bold=True)
    ws2.column_dimensions["A"].width = 24
    ws2.column_dimensions["B"].width = 20

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def state_badge(state: str) -> str:
    s = state.lower()
    if "work" in s:
        return f'<span class="badge-working">● Working</span>'
    elif "off" in s:
        return f'<span class="badge-offline">● Offline</span>'
    return f'<span class="badge-idle">● Idle</span>'


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <h1>🛵 Rider Performance Tracker</h1>
  <p>Upload dashboard screenshots → get accurate rider data & downloadable Excel report</p>
</div>
""", unsafe_allow_html=True)

# API key input
api_key = st.text_input(
    "Anthropic API Key",
    type="password",
    placeholder="sk-ant-...",
    help="Your key is never stored — used only for this session."
)

st.markdown('<div class="section-title">📤 Upload Screenshots</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Drag & drop rider dashboard screenshots",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    process_btn = st.button("🔍 Extract Rider Data", use_container_width=True)
with col_btn2:
    clear_btn = st.button("🗑️ Clear All", use_container_width=True)
with col_btn3:
    pass  # spacer

if clear_btn:
    st.session_state.all_riders = []
    st.rerun()

# ── Processing ────────────────────────────────────────────────────────────────
if process_btn:
    if not api_key:
        st.markdown('<div class="error-msg">❌ Please enter your Anthropic API key first.</div>', unsafe_allow_html=True)
    elif not uploaded_files:
        st.markdown('<div class="error-msg">❌ Please upload at least one screenshot.</div>', unsafe_allow_html=True)
    else:
        client = anthropic.Anthropic(api_key=api_key)
        progress = st.progress(0, text="Starting extraction…")
        new_riders = []

        for i, f in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Processing {f.name}…")
            try:
                f.seek(0)
                mt = f.type if f.type else "image/png"
                img_b64 = encode_image(f)
                riders = extract_riders_from_image(client, img_b64, mt)
                new_riders.extend(riders)
            except Exception as e:
                st.markdown(f'<div class="error-msg">⚠️ Error on {f.name}: {e}</div>', unsafe_allow_html=True)

        progress.progress(1.0, text="Done!")

        # Merge: update existing riders by ID, append new ones
        existing_ids = {r["rider_id"]: idx for idx, r in enumerate(st.session_state.all_riders)}
        for r in new_riders:
            rid = r.get("rider_id", "")
            if rid and rid in existing_ids:
                st.session_state.all_riders[existing_ids[rid]] = r
            else:
                st.session_state.all_riders.append(r)
                if rid:
                    existing_ids[rid] = len(st.session_state.all_riders) - 1

        if new_riders:
            st.markdown(f'<div class="success-msg">✅ Extracted {len(new_riders)} rider(s) from {len(uploaded_files)} image(s). Total in report: {len(st.session_state.all_riders)}</div>', unsafe_allow_html=True)

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.all_riders:
    riders = st.session_state.all_riders

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    working_count  = sum(1 for r in riders if "work" in str(r.get("rider_state","")).lower())
    utrs_all       = [r["utr"] for r in riders if isinstance(r.get("utr"), (int, float))]
    avg_utr        = round(sum(utrs_all)/len(utrs_all), 2) if utrs_all else 0
    total_accepted = sum(r.get("accepted", 0) or 0 for r in riders)
    total_deliveries = sum(r.get("deliveries", 0) or 0 for r in riders)

    for col, (label, val, sub) in zip(
        [m1, m2, m3, m4, m5],
        [
            ("Total Riders",    len(riders),       "in report"),
            ("Working Now",     working_count,      "active"),
            ("Avg UTR",         avg_utr,            "utilization"),
            ("Accepted Orders", total_accepted,     "real count"),
            ("System Deliveries", total_deliveries, "shown count"),
        ]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Rider cards ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏍️ Riders Detail</div>', unsafe_allow_html=True)

    for r in riders:
        badge = state_badge(r.get("rider_state", ""))
        curr  = r.get("currency", "")
        bal   = r.get("cash_balance", 0)
        st.markdown(f"""
        <div class="rider-card">
          <div>
            <div class="rider-name">{r.get('name', 'Unknown')}</div>
            <div class="rider-id">ID: {r.get('rider_id', '-')} &nbsp;|&nbsp; Session: {r.get('session', '-')}</div>
          </div>
          <div style="display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
            <div style="text-align:center">
              <div style="color:#8b8fa8; font-size:11px;">UTR</div>
              <div style="color:#fff; font-weight:700;">{r.get('utr', '-')}</div>
            </div>
            <div style="text-align:center">
              <div style="color:#8b8fa8; font-size:11px;">Accepted</div>
              <div style="color:#4ade80; font-weight:700;">{r.get('accepted', '-')}</div>
            </div>
            <div style="text-align:center">
              <div style="color:#8b8fa8; font-size:11px;">Sys. Deliveries</div>
              <div style="color:#fbbf24; font-weight:700;">{r.get('deliveries', '-')}</div>
            </div>
            <div style="text-align:center">
              <div style="color:#8b8fa8; font-size:11px;">Acceptance %</div>
              <div style="color:#fff; font-weight:700;">{r.get('acceptance_rate', '-')}%</div>
            </div>
            <div style="text-align:center">
              <div style="color:#8b8fa8; font-size:11px;">Balance</div>
              <div style="color:#fff; font-weight:700;">{curr} {bal}</div>
            </div>
            <div>{badge}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── DataFrame ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Full Data Table</div>', unsafe_allow_html=True)
    df = pd.DataFrame(riders)
    col_order = ["rider_id","name","session","utr","deliveries","accepted","stacked",
                 "acceptance_rate","cash_balance","currency","rider_state","location_area"]
    df = df.reindex(columns=[c for c in col_order if c in df.columns])
    df.columns = ["ID","Name","Session","UTR","Sys. Deliveries","Accepted","Stacked",
                  "Acceptance %","Balance","Currency","State","Location"][:len(df.columns)]
    st.dataframe(df, use_container_width=True, height=300)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⬇️ Download Report</div>', unsafe_allow_html=True)
    xlsx_bytes = build_excel(riders)
    fname = f"riders_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    st.download_button(
        label="📥 Download Excel Report",
        data=xlsx_bytes,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.markdown(f'<div class="info-msg">💡 Every time you upload new screenshots and click Extract, the report updates automatically. Existing riders are updated, new ones are added.</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #8b8fa8;">
      <div style="font-size: 48px;">🛵</div>
      <div style="font-size: 18px; font-weight: 600; color: #fff; margin-top: 16px;">No data yet</div>
      <div style="font-size: 14px; margin-top: 8px;">Upload screenshots above and click Extract to get started</div>
    </div>
    """, unsafe_allow_html=True)
