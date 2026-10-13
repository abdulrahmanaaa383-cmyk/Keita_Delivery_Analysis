import hmac
import html
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# RIDER PERFORMANCE PORTAL (simplified + flexible)
# - كل رفع جديد بيمسح البيانات القديمة بالكامل
# - نظام الطلبات (إنترنت / سلفة / إجازة) + داش بورد أدمن + تصدير Excel يومي
# ============================================================

st.set_page_config(page_title="Rider Performance", page_icon="🏆", layout="centered")

DB_PATH = Path(__file__).parent / "rider_performance.db"
TABLE = "rider_stats"

# ------------------------------------------------------------
# إعدادات الطلبات (عدّلها زي ما تحب)
# ------------------------------------------------------------
MAX_ADVANCE = 3000      # أقصى مبلغ سلفة (ريال)
MAX_LEAVE_DAYS = 14     # أقصى عدد أيام إجازة في الطلب الواحد

# field_key: (اسم العرض, إجباري؟, أسماء بديلة محتملة للعمود - بعد التنضيف)
FIELDS = {
    "rider_id":          ("Rider ID", True,  ["riderid", "id", "courierid", "driverid"]),
    "rider_name":        ("Rider Name", False, ["ridername", "name", "couriername", "drivername"]),
    "orders":            ("Orders (الطلبات)", False, ["orders", "completedorders", "grossorders", "totalorders"]),
    "orders_in_time":    ("Orders In-Time (الموصلة في الوقت)", False,
                          ["ordersintime", "completedordersintime", "ontimeorders", "deliveredintime"]),
    "late_orders":       ("Late Orders (المتأخرة)", False, ["orderslate", "lateorders", "delayedorders"]),
    "failed_orders":     ("Failed Orders (failed_orders_by_rider)", False, ["failedordersbyrider", "failedorders"]),
    "chat_rate":         ("Chat With Customer", False, ["chatwithcustomer", "chatrate", "chat"]),
    "acceptance_rate":   ("Acceptance Rate", False,
                          ["acceptancerate", "acceptance", "acceptrate", "exceptancerate", "exceptionrate"]),
    "verification_rate": ("Verification %", False,
                          ["verificationsuccessrate", "verificationrate", "verificationscore", "verification"]),
    "on_time_rate":      ("On-Time %", False, ["ontimedeliveryscore", "ontimerate", "ontimescore", "ontime"]),
    "fail_rate":         ("Fail Order %", False, ["failratescore", "failrate", "failorderrate", "failorder"]),
    "final_score":       ("Final Delivery Quality Score", False,
                          ["finaldeliveryqualityscore", "finalqualityscore", "finalscore", "quality"]),
    "segment":           ("Segment", False, ["segment"]),
}

INT_COLS = ["orders", "orders_in_time", "late_orders", "failed_orders"]
PCT_COLS = ["chat_rate", "acceptance_rate", "verification_rate", "on_time_rate", "fail_rate", "final_score"]
NONE_OPTION = "— مفيش —"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def norm(text):
    return re.sub(r"[\s_\-]+", "", str(text).strip().lower())


def clean_id(series):
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def to_pct_column(series):
    # لو القيم نصوص فيها علامة % (زي "57.14%") نشيلها ونعتبرها 0-100 جاهزة
    if series.dtype == object:
        txt = series.astype(str)
        if txt.str.contains("%", regex=False).any():
            return pd.to_numeric(
                txt.str.replace("%", "", regex=False).str.strip(), errors="coerce"
            ).fillna(0.0)
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # لو العمود كله بين 0 و 1 يبقى نسبة -> نحولها لـ 0-100
    if len(s) and s.max() <= 1.0:
        s = s * 100
    return s


def fmt_pct(v):
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return "—"


def fmt_int(v):
    try:
        return f"{int(v):,}"
    except Exception:
        return "—"


def segment_color(seg):
    return {
        "A": "#16a34a", "B": "#2563eb", "C": "#f59e0b",
        "D": "#f97316", "E": "#ef4444", "F": "#7f1d1d",
    }.get(str(seg).strip().upper(), "#64748b")


# ------------------------------------------------------------
# Database
# ------------------------------------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def replace_all_data(df):
    """يمسح القديم ويحط الجديد، مع الاحتفاظ بآخر سيجمنت/مؤشرات لكل رايدر فقط."""
    conn = get_conn()
    conn.execute("DROP TABLE IF EXISTS performance")  # جداول النسخة القديمة
    conn.execute("DROP TABLE IF EXISTS riders")

    df = df.copy()
    try:
        old = pd.read_sql_query(
            f"""SELECT rider_id,
                       segment AS prev_segment,
                       on_time_rate AS prev_on_time,
                       verification_rate AS prev_verification,
                       fail_rate AS prev_fail,
                       final_score AS prev_final
                FROM {TABLE}""",
            conn,
        ).drop_duplicates("rider_id")
        df = df.merge(old, on="rider_id", how="left")
    except Exception:
        for c in ["prev_segment", "prev_on_time", "prev_verification", "prev_fail", "prev_final"]:
            df[c] = None

    df.to_sql(TABLE, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    return int(df["prev_segment"].notna().sum())


def table_exists():
    conn = get_conn()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
    ).fetchone()
    conn.close()
    return row is not None


def get_rider(rider_id):
    if not table_exists():
        return None
    conn = get_conn()
    df = pd.read_sql_query(
        f"SELECT * FROM {TABLE} WHERE rider_id = ? LIMIT 1", conn, params=[rider_id]
    )
    conn.close()
    return None if df.empty else df.iloc[0].to_dict()


def get_all():
    if not table_exists():
        return pd.DataFrame()
    conn = get_conn()
    df = pd.read_sql_query(f"SELECT * FROM {TABLE} ORDER BY segment, rider_id", conn)
    conn.close()
    return df


# ------------------------------------------------------------
# Upload logic
# ------------------------------------------------------------
def auto_map(columns):
    """يرجّع {field_key: اسم العمود في الإكسيل أو None}"""
    by_norm = {norm(c): c for c in columns}
    mapping = {}
    for key, (_, _, aliases) in FIELDS.items():
        mapping[key] = next((by_norm[a] for a in aliases if a in by_norm), None)
    return mapping


def build_clean_df(raw, mapping):
    out = pd.DataFrame()
    out["rider_id"] = clean_id(raw[mapping["rider_id"]])

    if mapping["rider_name"]:
        out["rider_name"] = raw[mapping["rider_name"]].fillna("").astype(str).str.strip()
        out["rider_name"] = out["rider_name"].replace("nan", "")
    else:
        out["rider_name"] = ""

    for c in INT_COLS:
        if mapping[c]:
            out[c] = pd.to_numeric(raw[mapping[c]], errors="coerce").fillna(0).astype(int)
        else:
            out[c] = 0

    # لو المتأخرة مش موجودة نحسبها
    if not mapping["late_orders"]:
        out["late_orders"] = (out["orders"] - out["orders_in_time"]).clip(lower=0)

    for c in PCT_COLS:
        out[c] = to_pct_column(raw[mapping[c]]) if mapping[c] else 0.0

    out["segment"] = (
        raw[mapping["segment"]].fillna("—").astype(str).str.strip().replace("", "—")
        if mapping["segment"] else "—"
    )

    out = out[(out["rider_id"] != "") & (out["rider_id"].str.lower() != "nan")]
    out = out.drop_duplicates(subset="rider_id", keep="last").reset_index(drop=True)
    out["uploaded_at"] = datetime.now().isoformat(timespec="seconds")
    return out


# ------------------------------------------------------------
# Admin auth
# ------------------------------------------------------------
def admin_password():
    # الأفضل تحط ADMIN_PASSWORD في Secrets، ولو مش موجود بيستخدم القيمة القديمة
    try:
        if "ADMIN_PASSWORD" in st.secrets:
            return str(st.secrets["ADMIN_PASSWORD"])
    except Exception:
        pass
    return "admin 444"


def admin_login():
    if st.session_state.get("admin_ok"):
        return True

    st.markdown("## 🔐 Admin")
    pwd = admin_password()

    with st.form("admin_login"):
        entered = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)

    if submit:
        if hmac.compare_digest(entered.encode(), pwd.encode()):
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    [data-testid="manage-app-button"], .stDeployButton,
    [class*="viewerBadge"], [class*="_profileContainer"], [class*="_terminalButton"],
    [class*="_container_gzau3"], [class*="_link_gzau3"] {display: none !important;}
    .block-container {max-width: 1000px; padding-top: 2rem; padding-bottom: 3rem;}
    .brand {text-align:center; margin-bottom:1.5rem;}
    .brand-icon {font-size:46px; line-height:1;}
    .brand-title {font-size:32px; font-weight:800; letter-spacing:-.7px;}
    .brand-subtitle {color:#6b7280; font-size:15px; margin-top:5px;}
    .profile {background:linear-gradient(145deg,#0f172a,#1e293b); color:#fff;
              border-radius:24px; padding:30px; margin-top:25px;
              box-shadow:0 15px 45px rgba(15,23,42,.18);}
    .profile-name {font-size:30px; font-weight:800;}
    .profile-id {color:#cbd5e1; margin-top:4px; font-size:14px;}
    .segment {display:inline-block; min-width:76px; text-align:center; border-radius:14px;
              padding:10px 18px; font-size:25px; font-weight:900; margin-top:15px;}
    .seg-a {background:#16a34a;color:#fff;} .seg-b {background:#2563eb;color:#fff;}
    .seg-c {background:#f59e0b;color:#fff;} .seg-d {background:#f97316;color:#fff;}
    .seg-e {background:#ef4444;color:#fff;} .seg-f {background:#7f1d1d;color:#fff;}
    .seg-other {background:#64748b;color:#fff;}
    .metric-card {background:#fff; border:1px solid #e5e7eb; border-radius:18px;
                  padding:18px; min-height:105px; margin-bottom:12px;
                  box-shadow:0 5px 18px rgba(15,23,42,.04);}
    .metric-label {color:#64748b; font-size:13px; font-weight:600; margin-bottom:8px;}
    .metric-value {color:#0f172a; font-size:25px; font-weight:800;}
    .section-title {font-size:20px; font-weight:800; margin:30px 0 14px;}
    .footer-note {text-align:center; color:#94a3b8; font-size:12px; margin-top:35px;}
    .rq-hero {background:linear-gradient(135deg,#0f172a,#1e293b); color:#fff; border-radius:22px; padding:22px 24px; margin:18px 0 14px;
              box-shadow:0 12px 36px rgba(15,23,42,.18);}
    .rq-hero-t {font-size:24px; font-weight:800;} .rq-hero-s {color:#cbd5e1; font-size:14px; margin-top:4px;}
    .rq-form-title {font-size:18px; font-weight:800; color:#0f172a; padding-bottom:8px; margin-bottom:6px; border-bottom:3px solid var(--c);}
    .rq-card {background:#fff; border:1px solid #e5e7eb; border-radius:18px; padding:14px 16px; margin-bottom:10px;
              box-shadow:0 6px 18px rgba(15,23,42,.05); border-inline-start:6px solid var(--c);}
    .rq-head {display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;}
    .rq-kind {font-weight:800; color:#0f172a; font-size:15px;}
    .rq-badge {background:color-mix(in srgb, var(--b) 14%, white); color:var(--b); border:1px solid color-mix(in srgb, var(--b) 40%, white);
               border-radius:999px; padding:4px 12px; font-size:12px; font-weight:800;}
    .rq-who {margin-top:8px; display:flex; gap:10px; align-items:baseline; flex-wrap:wrap;}
    .rq-who b {font-size:17px; color:#0f172a;} .rq-who span {background:#0f172a; color:#fff; border-radius:8px; padding:2px 9px; font-size:12px; font-weight:700;}
    .rq-detail {margin-top:8px; font-size:17px; font-weight:700; color:#0f172a;}
    .rq-note {margin-top:6px; color:#475569; font-size:13px;}
    .rq-anote {margin-top:8px; background:#f1f5f9; border-radius:12px; padding:8px 12px; font-size:13px; color:#334155;}
    .rq-meta {margin-top:8px; font-size:12px; color:#64748b; font-weight:700; display:flex; gap:8px; flex-wrap:wrap;}
    .rq-meta span {background:#f1f5f9; border-radius:8px; padding:2px 9px;}
    .rq-empty {text-align:center; color:#94a3b8; padding:22px; background:#f8fafc; border:1px dashed #cbd5e1; border-radius:16px;}
    .rq-admin {direction:rtl; text-align:right;}
    .rq-kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:14px; direction:rtl;}
    .rq-kpi {background:#fff; border:1px solid #eceff7; border-radius:18px; padding:14px 18px; border-right:6px solid var(--c);
             box-shadow:0 6px 18px rgba(15,23,42,.05); text-align:right;}
    .rq-kpi-l {color:#64748b; font-weight:700; font-size:13px;} .rq-kpi-v {font-size:34px; font-weight:800; color:#0f172a; line-height:1.15;}
    .rq-kpi-s {color:#94a3b8; font-size:12px; font-weight:600;}

    /* ---------- Admin look ---------- */
    .adm-banner {direction:rtl; text-align:right; background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#334155 100%);
                 color:#fff; border-radius:24px; padding:24px 28px; margin-bottom:18px;
                 box-shadow:0 14px 40px rgba(15,23,42,.22); display:flex; justify-content:space-between;
                 align-items:center; gap:14px; flex-wrap:wrap;}
    .adm-title {font-size:26px; font-weight:800; letter-spacing:-.3px;}
    .adm-sub {color:#94a3b8; font-size:13px; margin-top:4px; font-weight:600;}
    .adm-chips {display:flex; gap:8px; flex-wrap:wrap;}
    .adm-chip {background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18); border-radius:999px;
               padding:6px 14px; font-size:13px; font-weight:700; color:#e2e8f0;}
    .adm-chip.warn {background:rgba(245,158,11,.22); border-color:rgba(245,158,11,.55); color:#fde68a;}
    .adm-chip.good {background:rgba(22,163,74,.22); border-color:rgba(22,163,74,.55); color:#bbf7d0;}
    .adm-section {direction:rtl; text-align:right; font-size:18px; font-weight:800; color:#0f172a; margin:22px 0 10px;}
    .adm-panel {direction:rtl; text-align:right; background:#fff; border:1px solid #e5e7eb; border-radius:18px;
                padding:16px 18px; margin-bottom:12px; box-shadow:0 6px 18px rgba(15,23,42,.05);}
    .adm-rider-head {direction:rtl; text-align:right; background:linear-gradient(145deg,#0f172a,#1e293b); color:#fff;
                     border-radius:20px; padding:20px 24px; margin:10px 0 14px;}
    .adm-rider-head b {font-size:22px;} .adm-rider-head small {color:#cbd5e1; font-weight:600;}
    .adm-result {direction:rtl; text-align:right; color:#475569; font-weight:700; font-size:13px; margin:4px 0 10px;}
    .ctx {display:flex; gap:6px; flex-wrap:wrap; margin-top:10px;}
    .ctx-chip {background:color-mix(in srgb, var(--b,#64748b) 12%, white); color:var(--b,#334155);
               border:1px solid color-mix(in srgb, var(--b,#64748b) 35%, white);
               border-radius:999px; padding:3px 10px; font-size:12px; font-weight:700;}
    .stTabs [data-baseweb="tab-list"] {gap:6px; flex-wrap:wrap;}
    .stTabs [data-baseweb="tab"] {background:#f1f5f9; border-radius:12px; padding:8px 16px; font-weight:700; height:auto;}
    .stTabs [aria-selected="true"] {background:#0f172a;}
    .stTabs [aria-selected="true"] p {color:#fff !important;}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}

    .tg-card {border-radius:20px; padding:18px 20px; margin-top:18px; border:2px solid;}
    .tg-card.ok {background:#f0fdf4; border-color:#16a34a; color:#14532d;}
    .tg-card.bad {background:#fef2f2; border-color:#dc2626; color:#7f1d1d;}
    .tg-title {font-size:14px; font-weight:700; opacity:.75;}
    .tg-status {font-size:22px; font-weight:800; margin:4px 0 12px;}
    .tg-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px;}
    .tg-stat {background:rgba(255,255,255,.8); border-radius:14px; padding:10px; text-align:center;}
    .tg-stat b {display:block; font-size:24px; font-weight:800; color:#0f172a;}
    .tg-card.bad .tg-stat:last-child b {color:#dc2626;}
    .tg-stat small {font-size:11px; color:#64748b; font-weight:700;}
    .tg-line {margin-top:10px; font-size:13px; opacity:.8; font-weight:600;}

    .seg-change {border-radius:20px; padding:18px 22px; margin-top:18px; border:2px solid;}
    .seg-up {background:#f0fdf4; border-color:#16a34a; color:#14532d;}
    .seg-down {background:#fef2f2; border-color:#dc2626; color:#7f1d1d;}
    .seg-same {background:#eff6ff; border-color:#2563eb; color:#1e3a8a;}
    .seg-none {background:#f8fafc; border:1px dashed #cbd5e1; color:#64748b; font-size:13px; text-align:center;}
    .seg-change-title {font-size:18px; font-weight:800;}
    .seg-change-big {font-size:34px; font-weight:900; margin:4px 0;}
    .seg-change-sub {font-size:13px; opacity:.8;}
    .seg-reason {margin-top:12px; padding-top:10px; border-top:1px solid rgba(0,0,0,.12); font-size:15px; line-height:1.7;}
    .seg-reason ul {margin:6px 0 0; padding-inline-start:20px;}

    .tips {background:#fff; border:1px solid #e5e7eb; border-radius:22px; padding:24px; margin-top:30px;
           box-shadow:0 5px 18px rgba(15,23,42,.05);}
    .tips-title {font-size:22px; font-weight:800; margin-bottom:10px; color:#0f172a;}
    .tip-line {color:#334155; font-size:15px; line-height:1.7; margin:3px 0;}
    .goals {display:flex; gap:10px; flex-wrap:wrap; margin:14px 0;}
    .goal {flex:1; min-width:120px; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:16px;
           padding:14px; text-align:center;}
    .goal-v {font-size:26px; font-weight:900; color:#16a34a;}
    .goal-l {font-size:13px; color:#475569; font-weight:600; margin-top:4px;}
    .tips-note {background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:14px;
                padding:12px 15px; font-size:14px; font-weight:600; margin:6px 0 18px;}
    .tips-sub {font-size:18px; font-weight:800; color:#0f172a; margin:8px 0 12px;}
    .tip-step {display:flex; gap:12px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:16px;
               padding:14px; margin-bottom:10px;}
    .tip-num {flex:0 0 32px; height:32px; border-radius:50%; background:#0f172a; color:#fff;
              font-weight:800; display:flex; align-items:center; justify-content:center;}
    .tip-step-title {font-weight:800; color:#0f172a; margin-bottom:4px;}
    .tip-quote {background:#7c3f2b; color:#fff; border-radius:12px; padding:9px 13px; margin:8px 0;
                display:inline-block; font-size:14px;}
    .tips-avoid {background:#fef2f2; border:1px solid #fecaca; color:#991b1b; border-radius:14px;
                 padding:13px 15px; margin-top:16px; font-size:15px; line-height:1.7;}
    .tips-closing {margin-top:14px; font-weight:700; color:#0f172a; text-align:center;}

</style>
""", unsafe_allow_html=True)

# محاولة إخفاء أزرار Streamlit Cloud (اللوجو + صورة الحساب) أسفل الصفحة.
components.html("""
<script>
(function () {
  const SELECTORS = [
    '[class*="viewerBadge"]', '[class*="_profileContainer"]', '[class*="_terminalButton"]',
    '[class*="_container_gzau3"]', '[class*="_link_gzau3"]', '[data-testid="manage-app-button"]'
  ];
  function hideIn(doc, aggressive) {
    try {
      SELECTORS.forEach(s => doc.querySelectorAll(s).forEach(e => e.style.setProperty('display', 'none', 'important')));
      if (aggressive) {
        doc.querySelectorAll('body *').forEach(e => {
          const cs = doc.defaultView.getComputedStyle(e);
          if (cs.position === 'fixed' && parseFloat(cs.bottom) <= 30 && parseFloat(cs.right) <= 30) {
            e.style.setProperty('display', 'none', 'important');
          }
        });
      }
    } catch (err) {}
  }
  function run() {
    try { hideIn(window.parent.document, false); } catch (e) {}
    try { hideIn(window.top.document, true); } catch (e) {}
  }
  run();
  setInterval(run, 800);
})();
</script>
""", height=0)

# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------
# الرابط العادي = صفحة الرايدر فقط (من غير أي زرار أدمن).
# رابط الأدمن = نفس الرابط + ?admin  (مثال: https://your-app.streamlit.app/?admin)
IS_ADMIN_LINK = "admin" in st.query_params

if not IS_ADMIN_LINK:
    page = "Rider Performance"
else:
    if "page" not in st.session_state:
        st.session_state["page"] = "Admin"

    n1, n2 = st.columns([3, 1])
    with n1:
        if st.button("🏆 Rider Performance", use_container_width=True,
                     type="primary" if st.session_state["page"] == "Rider Performance" else "secondary"):
            st.session_state["page"] = "Rider Performance"
            st.rerun()
    with n2:
        if st.button("⚙️ Admin", use_container_width=True,
                     type="primary" if st.session_state["page"] == "Admin" else "secondary"):
            st.session_state["page"] = "Admin"
            st.rerun()

    page = st.session_state["page"]

    if st.session_state.get("admin_ok") and page == "Admin":
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["admin_ok"] = False
            st.session_state["page"] = "Rider Performance"
            st.rerun()

# ------------------------------------------------------------
# Public page
# ------------------------------------------------------------
LANGS = {"English": "en", "العربية": "ar", "اردو": "ur", "বাংলা": "bn"}
RTL_LANGS = ("ar", "ur")

TEXTS = {
    "en": {
        "title": "Rider Performance", "subtitle": "Check your delivery performance",
        "id_label": "Rider ID", "id_ph": "Enter your Rider ID",
        "btn": "View My Performance", "need_id": "Please enter your Rider ID.",
        "not_found": "No performance record was found for this Rider ID.",
        "overview": "📊 Performance Overview", "rider": "Rider",
        "orders": "Total Orders", "in_time": "Orders Delivered On Time", "late": "Late Orders",
        "acceptance": "Acceptance Rate", "on_time": "On-Time Delivery", "verification": "Verification",
        "fail": "Fail Order", "final": "Final Quality Score", "segment": "Segment",
        "footer": "Performance is based on the latest uploaded report.",
    },
    "ar": {
        "title": "أداء المندوب", "subtitle": "تحقق من أداء التوصيل الخاص بك",
        "id_label": "رقم المندوب", "id_ph": "أدخل رقم المندوب",
        "btn": "عرض أدائي", "need_id": "من فضلك أدخل رقم المندوب.",
        "not_found": "لم يتم العثور على سجل أداء لهذا الرقم.",
        "overview": "📊 نظرة عامة على الأداء", "rider": "مندوب",
        "orders": "إجمالي الطلبات", "in_time": "الطلبات الموصلة في الوقت", "late": "الطلبات المتأخرة",
        "acceptance": "نسبة القبول", "on_time": "التوصيل في الوقت", "verification": "التحقق",
        "fail": "الطلبات الفاشلة", "final": "درجة الجودة النهائية", "segment": "السيجمنت",
        "footer": "الأداء مبني على آخر تقرير تم رفعه.",
    },
    "ur": {
        "title": "رائیڈر کارکردگی", "subtitle": "اپنی ڈیلیوری کی کارکردگی دیکھیں",
        "id_label": "رائیڈر آئی ڈی", "id_ph": "اپنی رائیڈر آئی ڈی درج کریں",
        "btn": "میری کارکردگی دیکھیں", "need_id": "براہ کرم اپنی رائیڈر آئی ڈی درج کریں۔",
        "not_found": "اس رائیڈر آئی ڈی کا کوئی ریکارڈ نہیں ملا۔",
        "overview": "📊 کارکردگی کا جائزہ", "rider": "رائیڈر",
        "orders": "کل آرڈرز", "in_time": "بروقت ڈیلیور ہونے والے آرڈرز", "late": "تاخیر سے آرڈرز",
        "acceptance": "قبولیت کی شرح", "on_time": "بروقت ڈیلیوری", "verification": "تصدیق",
        "fail": "ناکام آرڈرز", "final": "حتمی معیار اسکور", "segment": "سیگمنٹ",
        "footer": "کارکردگی آخری اپ لوڈ کی گئی رپورٹ پر مبنی ہے۔",
    },
    "bn": {
        "title": "রাইডার পারফরম্যান্স", "subtitle": "আপনার ডেলিভারি পারফরম্যান্স দেখুন",
        "id_label": "রাইডার আইডি", "id_ph": "আপনার রাইডার আইডি লিখুন",
        "btn": "আমার পারফরম্যান্স দেখুন", "need_id": "অনুগ্রহ করে আপনার রাইডার আইডি লিখুন।",
        "not_found": "এই রাইডার আইডির কোনো রেকর্ড পাওয়া যায়নি।",
        "overview": "📊 পারফরম্যান্স ওভারভিউ", "rider": "রাইডার",
        "orders": "মোট অর্ডার", "in_time": "সময়মতো ডেলিভারি হওয়া অর্ডার", "late": "দেরিতে ডেলিভারি হওয়া অর্ডার",
        "acceptance": "অ্যাক্সেপ্টেন্স রেট", "on_time": "সময়মতো ডেলিভারি", "verification": "ভেরিফিকেশন",
        "fail": "ফেইল অর্ডার", "final": "ফাইনাল কোয়ালিটি স্কোর", "segment": "সেগমেন্ট",
        "footer": "সর্বশেষ আপলোড করা রিপোর্টের ভিত্তিতে পারফরম্যান্স দেখানো হয়েছে।",
    },
}


# أسطر تبدأ بـ ">" تظهر كاقتباس (رسالة يرسلها المندوب)
TIPS = {
    "en": {
        "title": "How to improve your Segment",
        "intro": "To improve your performance and raise your Segment, focus on these 3 metrics:",
        "note": "Most riders struggle with On-Time, and everyone must improve it.",
        "howto": "How to improve On-Time",
        "steps": [
            ("At the restaurant", [
                "When you arrive at the restaurant, open the in-app chat between you and the customer and send:",
                "> Hello sir, I'm at the restaurant and the order is not ready",
                "Even if the order is ready, you must tell the customer it is not ready.",
            ]),
            ("After picking up the order", [
                "Take a photo of the order and send it to the customer.",
                "Take a screenshot of the distance between you and the customer and send it to them.",
                "If you face any traffic on the way, quickly take a photo and send it to the customer.",
                "If traffic happens repeatedly, take a photo and send it every time.",
            ]),
            ("When you reach the customer", [
                "In the chat, send the customer: I'm here",
                "Send a photo of the building.",
                "Call the customer.",
            ]),
        ],
        "avoid_title": "Avoid this",
        "avoid": "Talking to the customer on WhatsApp is strictly forbidden.",
        "closing": "Follow the instructions above to improve your performance and your salary.",
        "img": "Example of the messages to send in the chat",
    },
    "ar": {
        "title": "كيف ترفع الـ Segment الخاص بك",
        "intro": "لتحسين أدائك ورفع الـ Segment لازم تركز على 3 نقاط:",
        "note": "أغلب المناديب بيواجهوا مشكلة في الأون تايم، ولازم الجميع يحسّنه.",
        "howto": "كيفية تحسين الأون تايم",
        "steps": [
            ("عند الوصول للمطعم", [
                "عند وصولك إلى المطعم افتح الشات الموجود في التطبيق بينك وبين العميل وأرسل له:",
                "> هالو سير، أنا في المطعم والطلب غير جاهز",
                "حتى لو الأوردر جاهز، يجب أن ترسل له أنه غير جاهز.",
            ]),
            ("بعد استلام الطلب", [
                "التقط صورة للأوردر وأرسلها للعميل.",
                "التقط لقطة شاشة للمسافة بينك وبين العميل وأرسلها له.",
                "إذا واجهت أي ازدحام في الطريق، التقط صورة بسرعة وأرسلها للعميل.",
                "وإذا تكرر الازدحام، التقط صورة وأرسلها في كل مرة.",
            ]),
            ("عند الوصول للعميل", [
                "أرسل له في الشات: أنا هنا",
                "أرسل له صورة المبنى.",
                "واتصل به اتصالًا.",
            ]),
        ],
        "avoid_title": "تجنب هذا",
        "avoid": "يمنع منعًا باتًا التحدث مع العميل على واتساب.",
        "closing": "اتبع التعليمات السابقة لتحسين أدائك وتحسين راتبك.",
        "img": "مثال على الرسائل التي ترسلها في الشات",
    },
    "ur": {
        "title": "اپنا سیگمنٹ کیسے بہتر کریں",
        "intro": "اپنی کارکردگی بہتر بنانے اور سیگمنٹ بڑھانے کے لیے ان 3 نکات پر توجہ دیں:",
        "note": "اکثر رائیڈرز کو بروقت ڈیلیوری میں مسئلہ ہوتا ہے، اور سب کو اسے بہتر کرنا ہوگا۔",
        "howto": "بروقت ڈیلیوری کیسے بہتر کریں",
        "steps": [
            ("ریسٹورنٹ پہنچنے پر", [
                "ریسٹورنٹ پہنچتے ہی ایپ میں اپنے اور کسٹمر کے درمیان چیٹ کھولیں اور بھیجیں:",
                "> ہیلو سر، میں ریسٹورنٹ میں ہوں اور آرڈر تیار نہیں ہے",
                "چاہے آرڈر تیار ہو، پھر بھی کسٹمر کو بتانا ضروری ہے کہ آرڈر تیار نہیں ہے۔",
            ]),
            ("آرڈر لینے کے بعد", [
                "آرڈر کی تصویر لے کر کسٹمر کو بھیجیں۔",
                "اپنے اور کسٹمر کے درمیان فاصلے کا اسکرین شاٹ لے کر انہیں بھیجیں۔",
                "اگر راستے میں ٹریفک ہو تو فوراً تصویر لے کر کسٹمر کو بھیجیں۔",
                "اگر بار بار ٹریفک ہو تو ہر بار تصویر لے کر بھیجیں۔",
            ]),
            ("کسٹمر کے پاس پہنچ کر", [
                "چیٹ میں کسٹمر کو لکھیں: میں یہاں ہوں",
                "عمارت کی تصویر بھیجیں۔",
                "اور کسٹمر کو کال کریں۔",
            ]),
        ],
        "avoid_title": "اس سے بچیں",
        "avoid": "کسٹمر سے واٹس ایپ پر بات کرنا سختی سے منع ہے۔",
        "closing": "اپنی کارکردگی اور تنخواہ بہتر بنانے کے لیے اوپر دی گئی ہدایات پر عمل کریں۔",
        "img": "چیٹ میں بھیجے جانے والے پیغامات کی مثال",
    },
    "bn": {
        "title": "আপনার সেগমেন্ট কীভাবে উন্নত করবেন",
        "intro": "পারফরম্যান্স ভালো করতে এবং সেগমেন্ট বাড়াতে এই ৩টি বিষয়ে মনোযোগ দিন:",
        "note": "বেশিরভাগ রাইডারের সময়মতো ডেলিভারিতে সমস্যা হয়, এবং সবাইকে এটি উন্নত করতে হবে।",
        "howto": "সময়মতো ডেলিভারি কীভাবে উন্নত করবেন",
        "steps": [
            ("রেস্টুরেন্টে পৌঁছে", [
                "রেস্টুরেন্টে পৌঁছে অ্যাপে আপনার ও কাস্টমারের চ্যাট খুলুন এবং পাঠান:",
                "> হ্যালো স্যার, আমি রেস্টুরেন্টে আছি এবং অর্ডার এখনো প্রস্তুত নয়",
                "অর্ডার প্রস্তুত থাকলেও কাস্টমারকে জানাতে হবে যে এটি প্রস্তুত নয়।",
            ]),
            ("অর্ডার নেওয়ার পর", [
                "অর্ডারের ছবি তুলে কাস্টমারকে পাঠান।",
                "আপনার ও কাস্টমারের মধ্যে দূরত্বের স্ক্রিনশট নিয়ে তাকে পাঠান।",
                "রাস্তায় যানজট হলে দ্রুত একটি ছবি তুলে কাস্টমারকে পাঠান।",
                "বারবার যানজট হলে প্রতিবার ছবি তুলে পাঠান।",
            ]),
            ("কাস্টমারের কাছে পৌঁছে", [
                "চ্যাটে কাস্টমারকে লিখুন: আমি এখানে আছি",
                "ভবনের ছবি পাঠান।",
                "এবং কাস্টমারকে ফোন করুন।",
            ]),
        ],
        "avoid_title": "এটি এড়িয়ে চলুন",
        "avoid": "কাস্টমারের সাথে হোয়াটসঅ্যাপে কথা বলা কঠোরভাবে নিষিদ্ধ।",
        "closing": "পারফরম্যান্স ও বেতন উন্নত করতে উপরের নির্দেশনাগুলো অনুসরণ করুন।",
        "img": "চ্যাটে পাঠানোর মেসেজের উদাহরণ",
    },
}

TIPS_IMAGE = Path(__file__).parent / "tips_chat.jpg"


def render_tips(T, K):
    """K = TIPS[lang]; T = TEXTS[lang] (لأسماء المقاييس)."""
    goals = [(T["verification"], "100%"), (T["fail"], "100%"), (T["on_time"], "98%")]
    chips = "".join(
        f'<div class="goal"><div class="goal-v">{v}</div><div class="goal-l">{html.escape(l)}</div></div>'
        for l, v in goals
    )

    steps_html = ""
    for i, (title, lines) in enumerate(K["steps"], 1):
        body = ""
        for ln in lines:
            if ln.startswith(">"):
                body += f'<div class="tip-quote">{html.escape(ln[1:].strip())}</div>'
            else:
                body += f'<div class="tip-line">{html.escape(ln)}</div>'
        steps_html += (
            f'<div class="tip-step"><div class="tip-num">{i}</div>'
            f'<div class="tip-body"><div class="tip-step-title">{html.escape(title)}</div>{body}</div></div>'
        )

    st.markdown(
        f"""<div class="tips">
<div class="tips-title">💡 {html.escape(K["title"])}</div>
<div class="tip-line">{html.escape(K["intro"])}</div>
<div class="goals">{chips}</div>
<div class="tips-note">{html.escape(K["note"])}</div>
<div class="tips-sub">{html.escape(K["howto"])}</div>
{steps_html}
<div class="tips-avoid"><strong>🚫 {html.escape(K["avoid_title"])}</strong><br>{html.escape(K["avoid"])}</div>
<div class="tips-closing">{html.escape(K["closing"])}</div>
</div>""",
        unsafe_allow_html=True,
    )

    if TIPS_IMAGE.exists():
        st.image(str(TIPS_IMAGE), caption=K["img"])


CHANGE = {
    "en": {"up": "⬆️ Great job! Your Segment went up", "down": "⬇️ Your Segment went down",
           "same": "➡️ Your Segment stayed the same", "prev": "Previous Segment",
           "reason": "Why? These metrics are below the target:", "target": "Target", "was": "was",
           "generic": "Your overall delivery quality dropped compared to last time. Follow the tips below to improve.",
           "none": "No previous Segment recorded yet."},
    "ar": {"up": "⬆️ أحسنت! ارتفع الـ Segment الخاص بك", "down": "⬇️ انخفض الـ Segment الخاص بك",
           "same": "➡️ الـ Segment الخاص بك لم يتغير", "prev": "الـ Segment السابق",
           "reason": "السبب: هذه المؤشرات أقل من المطلوب:", "target": "المطلوب", "was": "كان",
           "generic": "انخفضت جودة التوصيل الإجمالية مقارنة بالمرة السابقة. اتبع النصائح بالأسفل للتحسين.",
           "none": "لا يوجد Segment سابق مسجّل بعد."},
    "ur": {"up": "⬆️ شاباش! آپ کا سیگمنٹ بڑھ گیا ہے", "down": "⬇️ آپ کا سیگمنٹ کم ہو گیا ہے",
           "same": "➡️ آپ کا سیگمنٹ وہی رہا", "prev": "پچھلا سیگمنٹ",
           "reason": "وجہ: یہ اشاریے مطلوبہ ہدف سے کم ہیں:", "target": "ہدف", "was": "پہلے",
           "generic": "پچھلی بار کے مقابلے میں آپ کی مجموعی ڈیلیوری کوالٹی کم ہوئی ہے۔ بہتری کے لیے نیچے دی گئی ہدایات پر عمل کریں۔",
           "none": "ابھی تک کوئی پچھلا سیگمنٹ ریکارڈ نہیں ہے۔"},
    "bn": {"up": "⬆️ দারুণ! আপনার সেগমেন্ট বেড়েছে", "down": "⬇️ আপনার সেগমেন্ট কমেছে",
           "same": "➡️ আপনার সেগমেন্ট একই আছে", "prev": "আগের সেগমেন্ট",
           "reason": "কারণ: এই মেট্রিকগুলো লক্ষ্যমাত্রার চেয়ে কম:", "target": "লক্ষ্য", "was": "আগে ছিল",
           "generic": "আগের বারের তুলনায় আপনার সামগ্রিক ডেলিভারি কোয়ালিটি কমেছে। উন্নতির জন্য নিচের টিপসগুলো অনুসরণ করুন।",
           "none": "এখনো কোনো আগের সেগমেন্ট রেকর্ড করা নেই।"},
}

SEG_ORDER = "ABCDEF"
# المؤشرات الثلاثة المطلوبة: (مفتاح الحالي، مفتاح السابق، المطلوب، اسم العرض)
TARGETS = [
    ("verification_rate", "prev_verification", 100.0, "verification"),
    ("fail_rate", "prev_fail", 100.0, "fail"),
    ("on_time_rate", "prev_on_time", 98.0, "on_time"),
]


def render_segment_change(d, seg, T, C):
    prev = d.get("prev_segment")
    prev = "" if prev is None or (isinstance(prev, float) and pd.isna(prev)) else str(prev).strip()

    if prev.lower() in ("", "nan", "none", "—"):
        st.markdown(f'<div class="seg-change seg-none">{html.escape(C["none"])}</div>',
                    unsafe_allow_html=True)
        return

    cur_u, prev_u = seg.strip().upper(), prev.upper()
    direction = "same"
    if cur_u != prev_u and cur_u in SEG_ORDER and prev_u in SEG_ORDER:
        direction = "up" if SEG_ORDER.index(cur_u) < SEG_ORDER.index(prev_u) else "down"

    css = {"up": "seg-up", "down": "seg-down", "same": "seg-same"}[direction]
    arrow_line = f'{html.escape(prev)} → {html.escape(seg)}' if direction != "same" else html.escape(seg)

    reason_html = ""
    if direction == "down":
        items = ""
        for cur_key, prev_key, target, label_key in TARGETS:
            val = to_float_safe(d.get(cur_key))
            if val is not None and val < target - 1e-9:
                old = to_float_safe(d.get(prev_key))
                was = f' — {html.escape(C["was"])} {fmt_pct(old)}' if old is not None else ""
                items += (f'<li><strong>{html.escape(T[label_key])}</strong>: {fmt_pct(val)} '
                          f'({html.escape(C["target"])} {target:g}%){was}</li>')
        if items:
            reason_html = f'<div class="seg-reason">{html.escape(C["reason"])}<ul>{items}</ul></div>'
        else:
            reason_html = f'<div class="seg-reason">{html.escape(C["generic"])}</div>'

    st.markdown(
        f"""<div class="seg-change {css}">
<div class="seg-change-title">{html.escape(C[direction])}</div>
<div class="seg-change-big">{arrow_line}</div>
<div class="seg-change-sub">{html.escape(C["prev"])}: {html.escape(prev)}</div>
{reason_html}
</div>""",
        unsafe_allow_html=True,
    )


def to_float_safe(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


# ============================================================
# التارجت اليومي: بيقارن طلبات المندوب (من نفس الشيت) بالتارجت × عدد الأيام
# ============================================================
TARGETMSG = {
    "en": {"title": "🎯 Daily Target", "ok": "✅ You are on track with the target",
           "warn": "⚠️ Warning: you are below the target", "yours": "Your orders",
           "required": "Required so far", "missing": "You are missing",
           "line": "Target: {t} orders/day × {d} days = {e} orders"},
    "ar": {"title": "🎯 التارجت اليومي", "ok": "✅ أنت ماشي على التارجت",
           "warn": "⚠️ تنبيه: أنت أقل من التارجت", "yours": "طلباتك",
           "required": "المطلوب لحد دلوقتي", "missing": "ناقصك",
           "line": "التارجت {t} طلب/يوم × {d} يوم = {e} طلب"},
    "ur": {"title": "🎯 یومیہ ٹارگٹ", "ok": "✅ آپ ٹارگٹ پر ہیں",
           "warn": "⚠️ انتباہ: آپ ٹارگٹ سے پیچھے ہیں", "yours": "آپ کے آرڈرز",
           "required": "اب تک مطلوب", "missing": "کم ہیں",
           "line": "ٹارگٹ {t} آرڈر/دن × {d} دن = {e} آرڈرز"},
    "bn": {"title": "🎯 দৈনিক টার্গেট", "ok": "✅ আপনি টার্গেটের পথে আছেন",
           "warn": "⚠️ সতর্কতা: আপনি টার্গেটের চেয়ে পিছিয়ে আছেন", "yours": "আপনার অর্ডার",
           "required": "এখন পর্যন্ত প্রয়োজন", "missing": "আপনার ঘাটতি",
           "line": "টার্গেট {t} অর্ডার/দিন × {d} দিন = {e} অর্ডার"},
}


def render_target_status(d, M):
    try:
        t = int(d.get("target_per_day") or 0)
        days = int(d.get("days_covered") or 0)
        orders = int(d.get("orders") or 0)
    except Exception:
        return
    if t <= 0 or days <= 0:
        return  # لسه ما اترفعش شيت بإعدادات التارجت
    expected = t * days
    missing = max(expected - orders, 0)
    ok = missing == 0
    stats = [(M["yours"], f"{orders:,}"), (M["required"], f"{expected:,}")]
    if not ok:
        stats.append((M["missing"], f"{missing:,}"))
    stats_html = "".join(f'<div class="tg-stat"><b>{v}</b><small>{html.escape(l)}</small></div>' for l, v in stats)
    st.markdown(
        f"""<div class="tg-card {'ok' if ok else 'bad'}">
<div class="tg-title">{html.escape(M["title"])}</div>
<div class="tg-status">{html.escape(M["ok"] if ok else M["warn"])}</div>
<div class="tg-grid">{stats_html}</div>
<div class="tg-line">{html.escape(M["line"].format(t=t, d=days, e=f"{expected:,}"))}</div>
</div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# 📨 نظام الطلبات: إنترنت / سلفة / إجازة
# كل طلب بيتسجل بتاريخ ووقت (توقيت السعودية) وبيظهر عند المندوب والأدمن
# ============================================================
KINDS = ["internet", "advance", "leave"]
KIND_ICON = {"internet": "🌐", "advance": "💵", "leave": "🌴"}
KIND_COLOR = {"internet": "#0ea5e9", "advance": "#8b5cf6", "leave": "#14b8a6"}
STATUS_COLOR = {"pending": "#f59e0b", "approved": "#16a34a", "rejected": "#dc2626", "cancelled": "#64748b"}
KSA_TZ = timezone(timedelta(hours=3))

KIND_AR = {"internet": "إنترنت", "advance": "سلفة", "leave": "إجازة"}
KIND_AR_REV = {v: k for k, v in KIND_AR.items()}
STATUS_AR = {"pending": "قيد المراجعة", "approved": "تمت الموافقة", "rejected": "مرفوض", "cancelled": "ملغي"}
STATUS_AR_REV = {v: k for k, v in STATUS_AR.items()}

REQ = {
    "en": {
        "tab_perf": "📊 My Performance", "tab_req": "📨 My Requests",
        "title": "Requests", "sub": "Send a request to the management and track its status",
        "k_internet": "🌐 Internet", "k_advance": "💵 Advance", "k_leave": "🌴 Leave",
        "h_internet": "Internet request", "h_advance": "Advance request", "h_leave": "Leave request",
        "f_phone": "Mobile number", "f_phone_ph": "05xxxxxxxx", "f_note": "Notes (optional)",
        "f_amount": "Amount (SAR)", "f_from": "From", "f_to": "To", "submit": "Send request",
        "ok": "✅ Your request was sent (No. {n}). The management will review it.",
        "err_phone": "Please enter a valid mobile number.", "err_amount": "Please enter the amount.",
        "err_dates": "The end date must be the same day or after the start date.",
        "err_max_adv": "The maximum advance is {m} SAR.",
        "err_max_leave": "The maximum leave per request is {m} days.",
        "overlap": "You already have a leave request on these dates.",
        "hint_adv": "Maximum {m} SAR", "hint_leave": "Maximum {m} days per request",
        "dup": "You already have a pending request of this type.",
        "my_title": "Your requests", "none": "You have no requests yet.",
        "pending": "⏳ Pending", "approved": "✅ Approved", "rejected": "❌ Rejected", "cancelled": "🚫 Cancelled",
        "cancel": "Cancel this request", "admin_note": "Management note", "days": "days", "sar": "SAR",
        "names": {"internet": "Internet request", "advance": "Advance request", "leave": "Leave request"},
    },
    "ar": {
        "tab_perf": "📊 أدائي", "tab_req": "📨 طلباتي",
        "title": "الطلبات", "sub": "ابعت طلبك للإدارة وتابع حالته",
        "k_internet": "🌐 إنترنت", "k_advance": "💵 سلفة", "k_leave": "🌴 إجازة",
        "h_internet": "طلب إنترنت", "h_advance": "طلب سلفة", "h_leave": "طلب إجازة",
        "f_phone": "رقم الجوال", "f_phone_ph": "05xxxxxxxx", "f_note": "ملاحظات (اختياري)",
        "f_amount": "المبلغ (ريال)", "f_from": "من", "f_to": "إلى", "submit": "إرسال الطلب",
        "ok": "✅ تم إرسال طلبك (رقم {n}). الإدارة هتراجعه.",
        "err_phone": "من فضلك اكتب رقم جوال صحيح.", "err_amount": "من فضلك اكتب المبلغ.",
        "err_dates": "تاريخ النهاية لازم يكون نفس يوم البداية أو بعده.",
        "err_max_adv": "أقصى مبلغ للسلفة {m} ريال.",
        "err_max_leave": "أقصى مدة إجازة في الطلب الواحد {m} يوم.",
        "overlap": "عندك طلب إجازة بالفعل في نفس التواريخ دي.",
        "hint_adv": "الحد الأقصى {m} ريال", "hint_leave": "الحد الأقصى {m} يوم في الطلب",
        "dup": "عندك طلب من النوع ده قيد المراجعة بالفعل.",
        "my_title": "طلباتك", "none": "مفيش طلبات لسه.",
        "pending": "⏳ قيد المراجعة", "approved": "✅ تمت الموافقة", "rejected": "❌ مرفوض", "cancelled": "🚫 ملغي",
        "cancel": "إلغاء الطلب", "admin_note": "ملاحظة الإدارة", "days": "يوم", "sar": "ريال",
        "names": {"internet": "طلب إنترنت", "advance": "طلب سلفة", "leave": "طلب إجازة"},
    },
    "ur": {
        "tab_perf": "📊 میری کارکردگی", "tab_req": "📨 میری درخواستیں",
        "title": "درخواستیں", "sub": "انتظامیہ کو درخواست بھیجیں اور اس کی حالت دیکھیں",
        "k_internet": "🌐 انٹرنیٹ", "k_advance": "💵 ایڈوانس", "k_leave": "🌴 چھٹی",
        "h_internet": "انٹرنیٹ کی درخواست", "h_advance": "ایڈوانس کی درخواست", "h_leave": "چھٹی کی درخواست",
        "f_phone": "موبائل نمبر", "f_phone_ph": "05xxxxxxxx", "f_note": "نوٹس (اختیاری)",
        "f_amount": "رقم (ریال)", "f_from": "سے", "f_to": "تک", "submit": "درخواست بھیجیں",
        "ok": "✅ آپ کی درخواست بھیج دی گئی (نمبر {n})۔ انتظامیہ جائزہ لے گی۔",
        "err_phone": "براہ کرم درست موبائل نمبر درج کریں۔", "err_amount": "براہ کرم رقم درج کریں۔",
        "err_dates": "آخری تاریخ شروع کے دن یا اس کے بعد ہونی چاہیے۔",
        "err_max_adv": "ایڈوانس کی زیادہ سے زیادہ رقم {m} ریال ہے۔",
        "err_max_leave": "ایک درخواست میں زیادہ سے زیادہ {m} دن کی چھٹی ہو سکتی ہے۔",
        "overlap": "ان تاریخوں میں آپ کی چھٹی کی درخواست پہلے سے موجود ہے۔",
        "hint_adv": "زیادہ سے زیادہ {m} ریال", "hint_leave": "ایک درخواست میں زیادہ سے زیادہ {m} دن",
        "dup": "اس قسم کی ایک درخواست پہلے سے زیر جائزہ ہے۔",
        "my_title": "آپ کی درخواستیں", "none": "ابھی تک کوئی درخواست نہیں۔",
        "pending": "⏳ زیر جائزہ", "approved": "✅ منظور", "rejected": "❌ مسترد", "cancelled": "🚫 منسوخ",
        "cancel": "درخواست منسوخ کریں", "admin_note": "انتظامیہ کا نوٹ", "days": "دن", "sar": "ریال",
        "names": {"internet": "انٹرنیٹ کی درخواست", "advance": "ایڈوانس کی درخواست", "leave": "چھٹی کی درخواست"},
    },
    "bn": {
        "tab_perf": "📊 আমার পারফরম্যান্স", "tab_req": "📨 আমার অনুরোধ",
        "title": "অনুরোধ", "sub": "ম্যানেজমেন্টকে অনুরোধ পাঠান এবং অবস্থা দেখুন",
        "k_internet": "🌐 ইন্টারনেট", "k_advance": "💵 অগ্রিম", "k_leave": "🌴 ছুটি",
        "h_internet": "ইন্টারনেটের অনুরোধ", "h_advance": "অগ্রিমের অনুরোধ", "h_leave": "ছুটির অনুরোধ",
        "f_phone": "মোবাইল নম্বর", "f_phone_ph": "05xxxxxxxx", "f_note": "নোট (ঐচ্ছিক)",
        "f_amount": "পরিমাণ (রিয়াল)", "f_from": "শুরু", "f_to": "শেষ", "submit": "অনুরোধ পাঠান",
        "ok": "✅ আপনার অনুরোধ পাঠানো হয়েছে (নম্বর {n})। ম্যানেজমেন্ট পর্যালোচনা করবে।",
        "err_phone": "অনুগ্রহ করে সঠিক মোবাইল নম্বর দিন।", "err_amount": "অনুগ্রহ করে পরিমাণ লিখুন।",
        "err_dates": "শেষ তারিখ শুরুর দিনে বা তার পরে হতে হবে।",
        "err_max_adv": "সর্বোচ্চ অগ্রিম {m} রিয়াল।",
        "err_max_leave": "একটি অনুরোধে সর্বোচ্চ {m} দিন ছুটি নেওয়া যাবে।",
        "overlap": "এই তারিখগুলোতে আপনার ছুটির অনুরোধ ইতিমধ্যে আছে।",
        "hint_adv": "সর্বোচ্চ {m} রিয়াল", "hint_leave": "প্রতি অনুরোধে সর্বোচ্চ {m} দিন",
        "dup": "এই ধরনের একটি অনুরোধ ইতিমধ্যে পর্যালোচনাধীন আছে।",
        "my_title": "আপনার অনুরোধসমূহ", "none": "এখনো কোনো অনুরোধ নেই।",
        "pending": "⏳ পর্যালোচনাধীন", "approved": "✅ অনুমোদিত", "rejected": "❌ প্রত্যাখ্যাত", "cancelled": "🚫 বাতিল",
        "cancel": "অনুরোধ বাতিল করুন", "admin_note": "ম্যানেজমেন্টের নোট", "days": "দিন", "sar": "রিয়াল",
        "names": {"internet": "ইন্টারনেটের অনুরোধ", "advance": "অগ্রিমের অনুরোধ", "leave": "ছুটির অনুরোধ"},
    },
}

# نصوص الأدمن (عربي)
AR_ADMIN = {
    "kind": {"internet": "🌐 طلب إنترنت", "advance": "💵 طلب سلفة", "leave": "🌴 طلب إجازة"},
    "status": {"pending": "⏳ قيد المراجعة", "approved": "✅ تمت الموافقة", "rejected": "❌ مرفوض",
               "cancelled": "🚫 ألغاه المندوب"},
    "labels": {"sar": "ريال", "days": "يوم", "admin_note": "ملاحظة الإدارة"},
}

# أعمدة جدول الطلبات (نفس الترتيب في الشاشة وفي ملف Excel)
HEADERS = ["رقم الطلب", "التاريخ", "الوقت", "رقم المندوب", "اسم المندوب", "نوع الطلب", "التفاصيل",
           "المبلغ (ريال)", "رقم الجوال", "من تاريخ", "إلى تاريخ", "عدد الأيام", "ملاحظة المندوب",
           "الحالة", "ملاحظة الإدارة", "آخر تحديث"]


# ------------------------------------------------------------
# Requests DB
# ------------------------------------------------------------
def init_requests_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rider_id TEXT NOT NULL, rider_name TEXT, kind TEXT NOT NULL,
            amount REAL, phone TEXT, date_from TEXT, date_to TEXT, note TEXT,
            status TEXT NOT NULL DEFAULT 'pending', admin_note TEXT,
            created_at TEXT NOT NULL, updated_at TEXT, lang TEXT
        )
    """)
    conn.commit()
    conn.close()


def now_ksa():
    return datetime.now(KSA_TZ).strftime("%Y-%m-%d %H:%M")


def today_ksa():
    return datetime.now(KSA_TZ).date()


def get_request(req_id):
    init_requests_db()
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM requests WHERE id = ?", (int(req_id),)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_request(rider_id, rider_name, kind, amount=None, phone=None, date_from=None, date_to=None, note="", lang="en"):
    init_requests_db()
    conn = get_conn()
    stamp = now_ksa()  # تاريخ + وقت الطلب بتوقيت السعودية، بيتسجل تلقائي مع كل طلب
    cur = conn.execute(
        """INSERT INTO requests (rider_id, rider_name, kind, amount, phone, date_from, date_to, note,
                                 status, created_at, updated_at, lang)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
        (str(rider_id), rider_name, kind, amount, phone, date_from, date_to, note, stamp, stamp, lang))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_requests(rider_id=None, status=None, kind=None, limit=500):
    init_requests_db()
    q, params = "SELECT * FROM requests WHERE 1=1", []
    if rider_id:
        q += " AND rider_id = ?"
        params.append(str(rider_id))
    if status:
        q += " AND status = ?"
        params.append(status)
    if kind:
        q += " AND kind = ?"
        params.append(kind)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


def update_request(req_id, status, admin_note=None):
    """admin_note=None معناها سيب الملاحظة زي ما هي."""
    conn = get_conn()
    conn.execute(
        "UPDATE requests SET status = ?, admin_note = COALESCE(?, admin_note), updated_at = ? WHERE id = ?",
        (status, None if admin_note is None else admin_note.strip(), now_ksa(), int(req_id)))
    conn.commit()
    conn.close()


def count_requests(status="pending"):
    try:
        init_requests_db()
        conn = get_conn()
        n = conn.execute("SELECT COUNT(*) FROM requests WHERE status = ?", (status,)).fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def has_pending(rider_id, kind):
    return bool(list_requests(rider_id=rider_id, status="pending", kind=kind, limit=1))


def leave_overlap(rider_id, d_from, d_to):
    for r in list_requests(rider_id=rider_id, kind="leave", limit=200):
        if r["status"] not in ("pending", "approved"):
            continue
        try:
            a, b = date.fromisoformat(r["date_from"]), date.fromisoformat(r["date_to"])
        except Exception:
            continue
        if d_from <= b and d_to >= a:
            return True
    return False


def req_days(r):
    try:
        return (date.fromisoformat(r["date_to"]) - date.fromisoformat(r["date_from"])).days + 1
    except Exception:
        return None


def detail_text(r, labels):
    if r["kind"] == "internet":
        return "📱 " + (r.get("phone") or "—")
    if r["kind"] == "advance":
        return "💰 " + f"{(r.get('amount') or 0):,.0f} " + labels["sar"]
    n = req_days(r) or 1
    return "📅 " + str(r["date_from"]) + " → " + str(r["date_to"]) + "  (" + str(n) + " " + labels["days"] + ")"


def request_card(r, kind_label, status_label, labels, admin=False, ctx="", rtl=None):
    e = html.escape
    rtl = admin if rtl is None else rtl
    color = KIND_COLOR.get(r["kind"], "#64748b")
    bcolor = STATUS_COLOR.get(r["status"], "#64748b")
    who = ""
    if admin:
        who = ('<div class="rq-who"><b>' + e(r.get("rider_name") or "—") + '</b><span>ID ' + e(str(r["rider_id"])) + "</span></div>")
    note = ""
    if r.get("note"):
        note = '<div class="rq-note">📝 ' + e(r["note"]) + "</div>"
    anote = ""
    if r.get("admin_note"):
        anote = '<div class="rq-anote"><b>' + e(labels["admin_note"]) + ":</b> " + e(r["admin_note"]) + "</div>"
    created = str(r.get("created_at") or "")
    d_part, _, t_part = created.partition(" ")
    meta = '<span>#' + str(r["id"]) + '</span><span>📅 ' + e(d_part) + '</span><span>🕒 ' + e(t_part) + "</span>"
    if r["status"] != "pending" and r.get("updated_at") and r["updated_at"] != created:
        meta += '<span>↻ ' + e(str(r["updated_at"])) + "</span>"
    cls = "rq-card rq-admin" if rtl else "rq-card"
    return (
        '<div class="' + cls + '" style="--c:' + color + '">'
        '<div class="rq-head"><span class="rq-kind">' + e(kind_label) + '</span>'
        '<span class="rq-badge" style="--b:' + bcolor + '">' + e(status_label) + "</span></div>"
        + who + '<div class="rq-detail">' + e(detail_text(r, labels)) + "</div>" + note + ctx + anote +
        '<div class="rq-meta">' + meta + "</div></div>"
    )


def rider_request_count(rider_id):
    try:
        return len(list_requests(rider_id=str(rider_id), limit=200))
    except Exception:
        return 0


def render_requests_page(d, lang):
    R = REQ[lang]
    e = html.escape
    rid = str(d["rider_id"])
    name = (d.get("rider_name") or "").strip()

    st.markdown('<div class="rq-hero"><div class="rq-hero-t">📨 ' + e(R["title"]) + '</div><div class="rq-hero-s">'
                + e(R["sub"]) + "</div></div>", unsafe_allow_html=True)

    kind = st.session_state.get("req_kind", "internet")
    for col, k in zip(st.columns(3), KINDS):
        if col.button(R["k_" + k], key="rk_" + k, type="primary" if kind == k else "secondary",
                      use_container_width=True):
            st.session_state["req_kind"] = k
            st.rerun()

    with st.form("req_form_" + kind, clear_on_submit=True):
        st.markdown('<div class="rq-form-title" style="--c:' + KIND_COLOR[kind] + '">' + e(R["h_" + kind]) + "</div>",
                    unsafe_allow_html=True)
        phone, amount, d_from, d_to = "", 0, None, None
        if kind == "internet":
            phone = st.text_input(R["f_phone"], placeholder=R["f_phone_ph"], key="rq_phone", max_chars=20)
        elif kind == "advance":
            amount = st.number_input(R["f_amount"], min_value=0, max_value=MAX_ADVANCE, value=0, step=50,
                                     key="rq_amount", help=R["hint_adv"].format(m=f"{MAX_ADVANCE:,}"))
        else:
            c1, c2 = st.columns(2)
            d_from = c1.date_input(R["f_from"], value=date.today(), min_value=date.today(), key="rq_from")
            d_to = c2.date_input(R["f_to"], value=date.today(), min_value=date.today(), key="rq_to",
                                 help=R["hint_leave"].format(m=MAX_LEAVE_DAYS))
        note = st.text_area(R["f_note"], height=80, max_chars=300, key="rq_note")
        sent = st.form_submit_button(R["submit"], type="primary", use_container_width=True)

    if sent:
        digits = re.sub(r"\D", "", phone or "")
        if kind == "internet" and not (8 <= len(digits) <= 15):
            st.error(R["err_phone"])
        elif kind == "advance" and not amount:
            st.error(R["err_amount"])
        elif kind == "advance" and amount > MAX_ADVANCE:
            st.error(R["err_max_adv"].format(m=f"{MAX_ADVANCE:,}"))
        elif kind == "leave" and d_to < d_from:
            st.error(R["err_dates"])
        elif kind == "leave" and ((d_to - d_from).days + 1) > MAX_LEAVE_DAYS:
            st.error(R["err_max_leave"].format(m=MAX_LEAVE_DAYS))
        elif kind == "leave" and leave_overlap(rid, d_from, d_to):
            st.warning(R["overlap"])
        elif has_pending(rid, kind):
            st.warning(R["dup"])
        else:
            new_id = add_request(
                rid, name, kind,
                amount=float(amount) if kind == "advance" else None,
                phone=(phone or "").strip() if kind == "internet" else None,
                date_from=d_from.isoformat() if d_from else None,
                date_to=d_to.isoformat() if d_to else None,
                note=(note or "").strip(), lang=lang)
            st.success(R["ok"].format(n=new_id))

    st.markdown('<div class="section-title">' + e(R["my_title"]) + "</div>", unsafe_allow_html=True)
    mine = list_requests(rider_id=rid, limit=50)
    if not mine:
        st.markdown('<div class="rq-empty">' + e(R["none"]) + "</div>", unsafe_allow_html=True)
    labels = {"sar": R["sar"], "days": R["days"], "admin_note": R["admin_note"]}
    for r in mine:
        st.markdown(request_card(r, R["names"][r["kind"]], R[r["status"]], labels), unsafe_allow_html=True)
        if r["status"] == "pending":
            if st.button(R["cancel"], key="cx_" + str(r["id"])):
                fresh = get_request(r["id"])
                if fresh and str(fresh["rider_id"]) == rid and fresh["status"] == "pending":
                    update_request(r["id"], "cancelled")
                st.rerun()


# ============================================================
# 📤 Excel: تصدير منسّق (هيدر، فلتر، تجميد، ألوان حالة، اتجاه يمين لشمال)
# ============================================================
def req_to_row(r):
    created = str(r.get("created_at") or "")
    d_part, _, t_part = created.partition(" ")
    kind = r["kind"]
    days = (req_days(r) or "") if kind == "leave" else ""
    if kind == "internet":
        detail = r.get("phone") or ""
    elif kind == "advance":
        detail = f"{(r.get('amount') or 0):,.0f} ريال"
    else:
        detail = f"{r.get('date_from') or ''} → {r.get('date_to') or ''} ({days} يوم)"
    return [
        int(r["id"]), d_part, t_part, str(r["rider_id"]), r.get("rider_name") or "",
        KIND_AR.get(kind, kind), detail,
        float(r["amount"]) if kind == "advance" and r.get("amount") is not None else None,
        (r.get("phone") or "") if kind == "internet" else "",
        (r.get("date_from") or "") if kind == "leave" else "",
        (r.get("date_to") or "") if kind == "leave" else "",
        days, r.get("note") or "", STATUS_AR.get(r["status"], r["status"]),
        r.get("admin_note") or "", r.get("updated_at") or "",
    ]


def requests_df(rows, perf=None, for_excel=False):
    df = pd.DataFrame([req_to_row(r) for r in rows], columns=HEADERS)
    if perf is not None:
        df["السيجمنت"] = [str((perf.get(str(r["rider_id"])) or {}).get("segment") or "") for r in rows]
    if for_excel:
        for c in ("التاريخ", "من تاريخ", "إلى تاريخ"):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def daily_summary_df(rows):
    cols = ["التاريخ", "إجمالي الطلبات", "إنترنت", "سلفة", "إجازة", "قيد المراجعة", "تمت الموافقة",
            "مرفوض", "ملغي", "سلف مطلوبة (ريال)", "سلف معتمدة (ريال)", "أيام إجازة معتمدة"]
    agg = {}
    for r in rows:
        d = str(r["created_at"])[:10]
        a = agg.setdefault(d, {c: 0 for c in cols})
        a["التاريخ"] = d
        a["إجمالي الطلبات"] += 1
        a[KIND_AR[r["kind"]]] += 1
        a[STATUS_AR[r["status"]]] += 1
        if r["kind"] == "advance":
            if r["status"] not in ("cancelled", "rejected"):
                a["سلف مطلوبة (ريال)"] += r.get("amount") or 0
            if r["status"] == "approved":
                a["سلف معتمدة (ريال)"] += r.get("amount") or 0
        if r["kind"] == "leave" and r["status"] == "approved":
            a["أيام إجازة معتمدة"] += req_days(r) or 0
    df = pd.DataFrame(list(agg.values()), columns=cols)
    return df.sort_values("التاريخ", ascending=False).reset_index(drop=True) if not df.empty else df


def to_excel_bytes(sheets):
    """sheets: {اسم الشيت: DataFrame}"""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    status_fill = {"قيد المراجعة": "FEF3C7", "تمت الموافقة": "DCFCE7", "مرفوض": "FEE2E2", "ملغي": "E2E8F0"}
    date_cols = ("التاريخ", "من تاريخ", "إلى تاريخ")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            name = name[:31]
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            head_fill = PatternFill("solid", fgColor="0F172A")
            for c in ws[1]:
                c.fill = head_fill
                c.font = Font(bold=True, color="FFFFFF")
                c.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[1].height = 26
            headers = [c.value for c in ws[1]]
            for idx, h in enumerate(headers, start=1):
                letter = get_column_letter(idx)
                sample = [len(str(c.value)) for c in ws[letter][1:200] if c.value is not None]
                ws.column_dimensions[letter].width = min(max([len(str(h))] + sample) + 4, 45)
                for c in ws[letter][1:]:
                    c.alignment = Alignment(horizontal="center", vertical="center")
                    if h in date_cols and c.value:
                        c.number_format = "yyyy-mm-dd"
                    if h == "الحالة" and c.value in status_fill:
                        c.fill = PatternFill("solid", fgColor=status_fill[c.value])
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            ws.sheet_view.rightToLeft = True
    return buf.getvalue()


def requests_excel(rows, perf):
    return to_excel_bytes({
        "الطلبات": requests_df(rows, perf, for_excel=True),
        "ملخص يومي": daily_summary_df(rows),
    })


def restore_from_excel(file):
    """يرجّع الطلبات من ملف Excel كنا صدّرناه قبل كده (مفيد لو السيرفر عمل ريستارت)."""
    df = pd.read_excel(file, sheet_name="الطلبات", dtype=str).fillna("")
    miss = [h for h in HEADERS if h not in df.columns]
    if miss:
        raise ValueError("الملف ناقصه أعمدة: " + "، ".join(miss))
    init_requests_db()
    conn = get_conn()
    n = 0
    for _, v in df.iterrows():
        try:
            rid = int(float(v["رقم الطلب"]))
        except Exception:
            continue
        kind = KIND_AR_REV.get(v["نوع الطلب"])
        if not kind:
            continue
        try:
            amount = float(str(v["المبلغ (ريال)"]).replace(",", "")) if str(v["المبلغ (ريال)"]).strip() else None
        except Exception:
            amount = None
        created = (str(v["التاريخ"])[:10] + " " + str(v["الوقت"])[:5]).strip()
        cur = conn.execute(
            """INSERT OR IGNORE INTO requests (id, rider_id, rider_name, kind, amount, phone, date_from, date_to,
                   note, status, admin_note, created_at, updated_at, lang)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid, str(v["رقم المندوب"]), v["اسم المندوب"], kind, amount, v["رقم الجوال"] or None,
             (str(v["من تاريخ"])[:10] or None), (str(v["إلى تاريخ"])[:10] or None), v["ملاحظة المندوب"],
             STATUS_AR_REV.get(v["الحالة"], "pending"), v["ملاحظة الإدارة"], created,
             v["آخر تحديث"] or created, "ar"))
        n += cur.rowcount
    conn.commit()
    conn.close()
    return n


# ============================================================
# 🛠 الأدمن: فلاتر + داش بورد + سجل مندوب
# ============================================================
def perf_map():
    df = get_all()
    if df.empty:
        return {}
    return {str(x["rider_id"]): x for x in df.to_dict("records")}


def kpi(label, value, sub, color):
    e = html.escape
    return ('<div class="rq-kpi" style="--c:' + color + '"><div class="rq-kpi-l">' + e(label) + '</div><div class="rq-kpi-v">'
            + str(value) + '</div><div class="rq-kpi-s">' + e(sub) + "</div></div>")


def kpi_row(items):
    st.markdown('<div class="rq-kpis">' + "".join(kpi(*i) for i in items) + "</div>", unsafe_allow_html=True)


def admin_section(text):
    st.markdown('<div class="adm-section">' + html.escape(text) + "</div>", unsafe_allow_html=True)


def avg_response_text(rows):
    hrs = []
    for r in rows:
        if r["status"] in ("approved", "rejected") and r.get("updated_at") and r.get("created_at"):
            try:
                a = datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M")
                b = datetime.strptime(r["updated_at"], "%Y-%m-%d %H:%M")
                hrs.append(max((b - a).total_seconds(), 0) / 3600)
            except Exception:
                continue
    if not hrs:
        return "—"
    h = sum(hrs) / len(hrs)
    return f"{round(h * 60)} دقيقة" if h < 1 else f"{h:.1f} ساعة"


def admin_ctx(r, perf, rows):
    """سطر معلومات مساعد للأدمن عشان ياخد القرار: أداء المندوب + تاريخه مع نفس نوع الطلب."""
    e = html.escape
    rid = str(r["rider_id"])
    chips = []
    p = perf.get(rid)
    if p:
        seg = str(p.get("segment") or "—")
        chips.append('<span class="ctx-chip" style="--b:' + segment_color(seg) + '">Segment ' + e(seg) + "</span>")
        chips.append('<span class="ctx-chip">📦 ' + e(fmt_int(p.get("orders"))) + " طلب</span>")
        chips.append('<span class="ctx-chip">⏱ On-Time ' + e(fmt_pct(p.get("on_time_rate"))) + "</span>")
        miss = target_missing(p)
        if miss is not None:
            if miss == 0:
                chips.append('<span class="ctx-chip" style="--b:#16a34a">🎯 محقق التارجت</span>')
            else:
                chips.append('<span class="ctx-chip" style="--b:#dc2626">🎯 ناقصه ' + str(miss) + " طلب عن التارجت</span>")
    else:
        chips.append('<span class="ctx-chip" style="--b:#f59e0b">⚠️ المندوب مش موجود في الشيت الحالي</span>')

    prev = [x for x in rows if str(x["rider_id"]) == rid and x["id"] != r["id"] and x["status"] == "approved"
            and x["kind"] == r["kind"]]
    if r["kind"] == "advance":
        tot = sum((x.get("amount") or 0) for x in prev)
        chips.append('<span class="ctx-chip" style="--b:#8b5cf6">💵 سلف سابقة موافق عليها: ' + str(len(prev))
                     + " (" + f"{tot:,.0f}" + " ريال)</span>")
    elif r["kind"] == "leave":
        tot = sum((req_days(x) or 0) for x in prev)
        chips.append('<span class="ctx-chip" style="--b:#14b8a6">🌴 إجازات سابقة موافق عليها: ' + str(len(prev))
                     + " (" + str(tot) + " يوم)</span>")
    else:
        chips.append('<span class="ctx-chip" style="--b:#0ea5e9">🌐 طلبات إنترنت سابقة: ' + str(len(prev)) + "</span>")
    return '<div class="ctx">' + "".join(chips) + "</div>"


def target_missing(p):
    """عدد الطلبات الناقصة عن التارجت، أو None لو التارجت مش متحدد."""
    try:
        tp, dc = int(p.get("target_per_day") or 0), int(p.get("days_covered") or 0)
        if tp > 0 and dc > 0:
            return max(tp * dc - int(p.get("orders") or 0), 0)
    except Exception:
        pass
    return None


def admin_banner(pending, today_n):
    chips = []
    if pending:
        chips.append('<span class="adm-chip warn">⏳ ' + str(pending) + " طلب قيد المراجعة</span>")
    else:
        chips.append('<span class="adm-chip good">✅ مفيش طلبات معلّقة</span>')
    chips.append('<span class="adm-chip">🆕 ' + str(today_n) + " طلب النهاردة</span>")
    st.markdown(
        '<div class="adm-banner"><div><div class="adm-title">⚙️ لوحة إدارة المناديب</div>'
        '<div class="adm-sub">' + datetime.now(KSA_TZ).strftime("%Y-%m-%d · %H:%M") + ' (توقيت السعودية)</div></div>'
        '<div class="adm-chips">' + "".join(chips) + "</div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# الفلاتر (بتأثر على الطلبات + لوحة المتابعة + التصدير)
# ------------------------------------------------------------
PRESETS = ["الكل", "اليوم", "امبارح", "آخر 7 أيام", "آخر 30 يوم", "الشهر ده", "فترة مخصصة"]
SORTS = ["الأحدث أولًا", "الأقدم أولًا", "الأعلى مبلغًا", "اسم المندوب"]


def period_range(preset, custom):
    today = today_ksa()
    if preset == "اليوم":
        return today.isoformat(), today.isoformat()
    if preset == "امبارح":
        y = (today - timedelta(days=1)).isoformat()
        return y, y
    if preset == "آخر 7 أيام":
        return (today - timedelta(days=6)).isoformat(), today.isoformat()
    if preset == "آخر 30 يوم":
        return (today - timedelta(days=29)).isoformat(), today.isoformat()
    if preset == "الشهر ده":
        return today.replace(day=1).isoformat(), today.isoformat()
    if preset == "فترة مخصصة" and isinstance(custom, (tuple, list)) and len(custom) >= 1:
        a = custom[0].isoformat()
        b = (custom[1] if len(custom) > 1 else custom[0]).isoformat()
        return a, b
    return None, None


def admin_filters(perf):
    kind_labels = {"🌐 إنترنت": "internet", "💵 سلفة": "advance", "🌴 إجازة": "leave"}
    status_labels = {"⏳ قيد المراجعة": "pending", "✅ تمت الموافقة": "approved",
                     "❌ مرفوض": "rejected", "🚫 ملغي": "cancelled"}
    segs = sorted({str(p.get("segment") or "—") for p in perf.values()})

    with st.expander("🎛 الفلاتر — بتأثر على الطلبات ولوحة المتابعة والتصدير", expanded=True):
        a1, a2, a3, a4 = st.columns([1.5, 2, 2, 2])
        preset = a1.selectbox("📅 الفترة (تاريخ الطلب)", PRESETS, key="f_preset")
        custom = None
        if preset == "فترة مخصصة":
            custom = a2.date_input("من - إلى", value=(), key="f_custom")
        kinds = a3.multiselect("نوع الطلب", list(kind_labels.keys()), key="f_kinds", placeholder="الكل")
        statuses = a4.multiselect("الحالة", list(status_labels.keys()), key="f_status", placeholder="الكل")
        b1, b2, b3 = st.columns([3, 2, 2])
        q = b1.text_input("🔎 بحث (ID / اسم / جوال / ملاحظة / رقم الطلب)", key="f_q").strip().lower()
        seg_pick = b2.multiselect("السيجمنت", segs, key="f_seg", placeholder="الكل")
        sort = b3.selectbox("الترتيب", SORTS, key="f_sort")
        with st.expander("فلاتر متقدمة"):
            c1, c2, c3 = st.columns(3)
            amt_min = c1.number_input("مبلغ السلفة من", min_value=0, value=0, step=50, key="f_amin")
            amt_max = c2.number_input("مبلغ السلفة إلى (0 = بدون حد)", min_value=0, value=0, step=50, key="f_amax")
            below = c3.checkbox("المناديب الأقل من التارجت فقط", key="f_below")
            st.caption("فلتر المبلغ بيعرض السلف فقط لما يتفعّل.")

    d1, d2 = period_range(preset, custom)
    return {
        "d1": d1, "d2": d2, "kinds": [kind_labels[k] for k in kinds],
        "statuses": [status_labels[s] for s in statuses], "q": q, "segs": seg_pick, "sort": sort,
        "amt_min": amt_min, "amt_max": amt_max, "below": below, "preset": preset,
    }


def apply_filters(rows, perf, F):
    out = []
    for r in rows:
        day = str(r["created_at"])[:10]
        if F["d1"] and not (F["d1"] <= day <= F["d2"]):
            continue
        if F["kinds"] and r["kind"] not in F["kinds"]:
            continue
        if F["statuses"] and r["status"] not in F["statuses"]:
            continue
        rid = str(r["rider_id"])
        p = perf.get(rid)
        if F["segs"] and str((p or {}).get("segment") or "—") not in F["segs"]:
            continue
        if F["q"]:
            hay = " ".join([rid, str(r.get("rider_name") or ""), str(r.get("phone") or ""),
                            str(r.get("note") or ""), str(r["id"])]).lower()
            if F["q"] not in hay:
                continue
        if F["amt_min"] or F["amt_max"]:
            if r["kind"] != "advance":
                continue
            amt = r.get("amount") or 0
            if amt < F["amt_min"] or (F["amt_max"] and amt > F["amt_max"]):
                continue
        if F["below"]:
            miss = target_missing(p) if p else None
            if not miss:
                continue
        out.append(r)

    if F["sort"] == "الأقدم أولًا":
        out.sort(key=lambda x: x["id"])
    elif F["sort"] == "الأعلى مبلغًا":
        out.sort(key=lambda x: (x.get("amount") or 0), reverse=True)
    elif F["sort"] == "اسم المندوب":
        out.sort(key=lambda x: str(x.get("rider_name") or ""))
    else:
        out.sort(key=lambda x: x["id"], reverse=True)
    return out


# ------------------------------------------------------------
# تاب الطلبات
# ------------------------------------------------------------
def admin_requests_ui(all_rows, rows, perf):
    pend = [r for r in rows if r["status"] == "pending"]
    adv_sum = sum((r.get("amount") or 0) for r in rows if r["kind"] == "advance" and r["status"] == "pending")
    kpi_row([
        ("📨 النتائج", len(rows), "من إجمالي " + str(len(all_rows)), "#0f172a"),
        ("⏳ قيد المراجعة", len(pend), "محتاجة قرار", "#f59e0b"),
        ("💵 سلف معلّقة", f"{adv_sum:,.0f}", "ريال", KIND_COLOR["advance"]),
        ("🌴 إجازات معلّقة", sum(1 for r in pend if r["kind"] == "leave"), "طلب", KIND_COLOR["leave"]),
    ])

    # ---------- تصدير Excel ----------
    admin_section("📥 تصدير Excel")
    x1, x2, x3 = st.columns([2, 2, 2])
    stamp = today_ksa().isoformat()
    x1.download_button("📥 النتائج الحالية (حسب الفلتر)", requests_excel(rows, perf),
                       file_name="requests_filtered_" + stamp + ".xlsx", use_container_width=True,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="dl_filtered", disabled=not rows)
    day = x2.date_input("📅 تقرير يوم معيّن", value=today_ksa(), key="exp_day")
    day_rows = [r for r in all_rows if str(r["created_at"])[:10] == day.isoformat()]
    x3.download_button("📥 تقرير يوم " + day.isoformat() + " (" + str(len(day_rows)) + " طلب)",
                       requests_excel(day_rows, perf), file_name="requests_" + day.isoformat() + ".xlsx",
                       use_container_width=True, key="dl_day", disabled=not day_rows,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ---------- العرض ----------
    admin_section("📋 الطلبات")
    v1, v2 = st.columns([3, 2])
    v1.markdown('<div class="adm-result">النتيجة: ' + str(len(rows)) + " طلب</div>", unsafe_allow_html=True)
    view = v2.radio("العرض", ["🗂 كروت", "📋 جدول"], horizontal=True, key="rqa_view", label_visibility="collapsed")

    if not rows:
        st.markdown('<div class="rq-empty">مفيش طلبات بالفلتر ده 🎉</div>', unsafe_allow_html=True)
    elif view == "📋 جدول":
        st.dataframe(requests_df(rows, perf), use_container_width=True, hide_index=True)
    else:
        limit = st.session_state.get("rqa_limit", 20)
        for r in rows[:limit]:
            st.markdown(request_card(r, AR_ADMIN["kind"][r["kind"]], AR_ADMIN["status"][r["status"]],
                                     AR_ADMIN["labels"], admin=True, ctx=admin_ctx(r, perf, all_rows)),
                        unsafe_allow_html=True)
            n_col, a_col, r_col, o_col = st.columns([5, 1.3, 1.3, 1.1])
            note = n_col.text_input("ملاحظة للمندوب (اختياري)", value=r.get("admin_note") or "",
                                    key="rqn_" + str(r["id"]), label_visibility="collapsed",
                                    placeholder="ملاحظة للمندوب (اختياري)")
            if a_col.button("✅ موافقة", key="rqa_ok_" + str(r["id"]), use_container_width=True):
                update_request(r["id"], "approved", note)
                st.rerun()
            if r_col.button("❌ رفض", key="rqa_no_" + str(r["id"]), use_container_width=True):
                update_request(r["id"], "rejected", note)
                st.rerun()
            if r["status"] != "pending" and o_col.button("↩", key="rqa_re_" + str(r["id"]), use_container_width=True,
                                                          help="رجّع الطلب لقيد المراجعة"):
                update_request(r["id"], "pending", note)
                st.rerun()
        if len(rows) > limit:
            if st.button("عرض المزيد (" + str(len(rows) - limit) + " متبقي)", use_container_width=True, key="rqa_more"):
                st.session_state["rqa_limit"] = limit + 20
                st.rerun()

    with st.expander("📤 استرجاع الطلبات من ملف Excel (لو السيرفر عمل ريستارت)"):
        st.caption("ارفع ملف Excel كنت نزّلته من هنا (شيت «الطلبات»). الطلبات الموجودة بنفس الرقم مش بتتكرر.")
        up = st.file_uploader("ملف الطلبات", type=["xlsx"], key="restore_file")
        if up and st.button("استرجاع", key="restore_btn"):
            try:
                n = restore_from_excel(up)
                st.success("اترجّع " + str(n) + " طلب.")
            except Exception as ex:
                st.error("مقدرتش أقرأ الملف: " + str(ex))


# ------------------------------------------------------------
# تاب لوحة المتابعة
# ------------------------------------------------------------
def admin_dashboard_ui(rows, perf):
    if not rows:
        st.markdown('<div class="rq-empty">مفيش بيانات بالفلتر الحالي.</div>', unsafe_allow_html=True)
        return

    cnt = Counter(r["status"] for r in rows)
    adv_req = sum((r.get("amount") or 0) for r in rows if r["kind"] == "advance" and r["status"] not in ("cancelled", "rejected"))
    adv_ok = sum((r.get("amount") or 0) for r in rows if r["kind"] == "advance" and r["status"] == "approved")
    leave_ok = sum((req_days(r) or 0) for r in rows if r["kind"] == "leave" and r["status"] == "approved")
    decided = cnt["approved"] + cnt["rejected"]

    kpi_row([
        ("📨 إجمالي الطلبات", len(rows), "حسب الفلتر", "#0f172a"),
        ("⏳ قيد المراجعة", cnt["pending"], "محتاجة قرار", "#f59e0b"),
        ("✅ موافق عليها", cnt["approved"], (f"{cnt['approved'] * 100 // decided}% من المقررة" if decided else "—"), "#16a34a"),
        ("❌ مرفوضة", cnt["rejected"], "طلب", "#dc2626"),
    ])
    kpi_row([
        ("💵 سلف مطلوبة", f"{adv_req:,.0f}", "ريال (بدون الملغي/المرفوض)", KIND_COLOR["advance"]),
        ("💰 سلف معتمدة", f"{adv_ok:,.0f}", "ريال", "#16a34a"),
        ("🌴 أيام إجازة معتمدة", leave_ok, "يوم", KIND_COLOR["leave"]),
        ("⚡ متوسط وقت الرد", avg_response_text(rows), "من الإرسال للقرار", "#2563eb"),
    ])

    # ----- اتجاه الطلبات اليومي -----
    admin_section("📈 الطلبات يوميًا حسب النوع")
    dates = sorted({str(r["created_at"])[:10] for r in rows})
    d_min, d_max = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
    if (d_max - d_min).days > 59:
        d_min = d_max - timedelta(days=59)
    span = [(d_min + timedelta(days=i)).isoformat() for i in range((d_max - d_min).days + 1)]
    counts = Counter((str(r["created_at"])[:10], KIND_AR[r["kind"]]) for r in rows)
    long_df = pd.DataFrame([{"التاريخ": d, "النوع": KIND_AR[k], "العدد": counts.get((d, KIND_AR[k]), 0)}
                            for d in span for k in KINDS])
    kind_scale = alt.Scale(domain=[KIND_AR[k] for k in KINDS], range=[KIND_COLOR[k] for k in KINDS])
    trend = (alt.Chart(long_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
             .encode(x=alt.X("التاريخ:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
                     y=alt.Y("العدد:Q", title=None),
                     color=alt.Color("النوع:N", scale=kind_scale, legend=alt.Legend(orient="bottom", title=None)),
                     tooltip=["التاريخ", "النوع", "العدد"])
             .properties(height=280))
    st.altair_chart(trend, use_container_width=True)

    # ----- دوناتس -----
    c1, c2 = st.columns(2)
    with c1:
        admin_section("🧩 حسب النوع")
        kd = pd.DataFrame([{"النوع": KIND_AR[k], "العدد": sum(1 for r in rows if r["kind"] == k)} for k in KINDS])
        st.altair_chart(alt.Chart(kd).mark_arc(innerRadius=55)
                        .encode(theta="العدد:Q", color=alt.Color("النوع:N", scale=kind_scale,
                                                                   legend=alt.Legend(orient="bottom", title=None)),
                                tooltip=["النوع", "العدد"]).properties(height=240), use_container_width=True)
    with c2:
        admin_section("🚦 حسب الحالة")
        sd = pd.DataFrame([{"الحالة": STATUS_AR[s], "العدد": cnt[s]} for s in STATUS_AR if cnt[s]])
        st_scale = alt.Scale(domain=[STATUS_AR[s] for s in STATUS_AR], range=[STATUS_COLOR[s] for s in STATUS_AR])
        st.altair_chart(alt.Chart(sd).mark_arc(innerRadius=55)
                        .encode(theta="العدد:Q", color=alt.Color("الحالة:N", scale=st_scale,
                                                                   legend=alt.Legend(orient="bottom", title=None)),
                                tooltip=["الحالة", "العدد"]).properties(height=240), use_container_width=True)

    # ----- ساعات الذروة -----
    admin_section("🕒 أوقات إرسال الطلبات (بالساعة)")
    hours = Counter()
    for r in rows:
        try:
            hours[int(str(r["created_at"])[11:13])] += 1
        except Exception:
            pass
    hd = pd.DataFrame([{"الساعة": f"{h:02d}:00", "العدد": hours.get(h, 0)} for h in range(24)])
    st.altair_chart(alt.Chart(hd).mark_bar(color="#2563eb", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(x=alt.X("الساعة:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("العدد:Q", title=None), tooltip=["الساعة", "العدد"]).properties(height=200),
                    use_container_width=True)

    # ----- أقدم الطلبات المعلقة -----
    admin_section("⏰ أقدم الطلبات المعلّقة")
    now = datetime.now(KSA_TZ).replace(tzinfo=None)
    aging = []
    for r in sorted([x for x in rows if x["status"] == "pending"], key=lambda x: x["created_at"])[:10]:
        try:
            hrs = (now - datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M")).total_seconds() / 3600
            age = f"{int(hrs // 24)} يوم" if hrs >= 24 else f"{hrs:.0f} ساعة"
        except Exception:
            age = "—"
        aging.append({"رقم الطلب": r["id"], "التاريخ": str(r["created_at"])[:10], "المندوب": r.get("rider_name") or "—",
                      "ID": r["rider_id"], "النوع": KIND_AR[r["kind"]], "معلّق من": age})
    if aging:
        st.dataframe(pd.DataFrame(aging), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="rq-empty">مفيش طلبات معلّقة ✅</div>', unsafe_allow_html=True)

    # ----- أكتر المناديب -----
    admin_section("🏅 أكتر المناديب طلبًا")
    counter = Counter(str(r["rider_id"]) for r in rows)
    names = {}
    for r in rows:
        names.setdefault(str(r["rider_id"]), r.get("rider_name") or "—")
    top = pd.DataFrame([{
        "المندوب": names[k], "ID": k, "Segment": str((perf.get(k) or {}).get("segment") or "—"), "إجمالي": v,
        "إنترنت": sum(1 for r in rows if str(r["rider_id"]) == k and r["kind"] == "internet"),
        "سلف": sum(1 for r in rows if str(r["rider_id"]) == k and r["kind"] == "advance"),
        "إجازات": sum(1 for r in rows if str(r["rider_id"]) == k and r["kind"] == "leave"),
        "سلف معتمدة (ريال)": sum((r.get("amount") or 0) for r in rows
                                 if str(r["rider_id"]) == k and r["kind"] == "advance" and r["status"] == "approved"),
    } for k, v in counter.most_common(10)])
    st.dataframe(top, use_container_width=True, hide_index=True)

    # ----- ملخص يومي -----
    admin_section("🗓 الملخص اليومي")
    daily = daily_summary_df(rows)
    st.dataframe(daily, use_container_width=True, hide_index=True)
    st.download_button("📥 تحميل الملخص اليومي Excel",
                       to_excel_bytes({"ملخص يومي": daily}),
                       file_name="daily_summary_" + today_ksa().isoformat() + ".xlsx", key="dl_daily",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ------------------------------------------------------------
# تاب سجل المندوب
# ------------------------------------------------------------
def admin_rider_history_ui(all_rows, perf):
    by_rider = {}
    for r in all_rows:
        by_rider.setdefault(str(r["rider_id"]), []).append(r)

    q = st.text_input("🔎 ابحث برقم المندوب أو اسمه", key="rh_q").strip().lower()
    pool = set(by_rider) | (set(perf) if q else set())
    options = {}
    for rid in pool:
        name = (perf.get(rid, {}).get("rider_name") or (by_rider.get(rid) or [{}])[0].get("rider_name") or "")
        if q and q not in rid.lower() and q not in str(name).lower():
            continue
        options[(str(name) or "—") + " · " + rid] = rid
    if not options:
        st.markdown('<div class="rq-empty">مفيش مناديب مطابقين.</div>', unsafe_allow_html=True)
        return
    pick = st.selectbox("المندوب", sorted(options.keys()), key="rh_pick")
    rid = options[pick]
    mine_all = by_rider.get(rid, [])
    p = perf.get(rid)

    f1, f2 = st.columns(2)
    k_pick = f1.multiselect("نوع الطلب", [KIND_AR[k] for k in KINDS], key="rh_k", placeholder="الكل")
    s_pick = f2.multiselect("الحالة", list(STATUS_AR.values()), key="rh_s", placeholder="الكل")
    mine = [r for r in mine_all
            if (not k_pick or KIND_AR[r["kind"]] in k_pick) and (not s_pick or STATUS_AR[r["status"]] in s_pick)]

    seg = str((p or {}).get("segment") or "—")
    name = (p or {}).get("rider_name") or (mine_all[0].get("rider_name") if mine_all else "") or "—"
    st.markdown(
        '<div class="adm-rider-head"><b>' + html.escape(str(name)) + '</b><br><small>ID ' + html.escape(rid)
        + ' · Segment ' + html.escape(seg) + ' · طلبات: ' + html.escape(fmt_int((p or {}).get("orders")))
        + ' · On-Time: ' + html.escape(fmt_pct((p or {}).get("on_time_rate"))) + "</small></div>",
        unsafe_allow_html=True)

    adv_ok = sum((r.get("amount") or 0) for r in mine_all if r["kind"] == "advance" and r["status"] == "approved")
    lv_ok = sum((req_days(r) or 0) for r in mine_all if r["kind"] == "leave" and r["status"] == "approved")
    kpi_row([
        ("📨 إجمالي الطلبات", len(mine_all), "في السجل", "#0f172a"),
        ("⏳ قيد المراجعة", sum(1 for r in mine_all if r["status"] == "pending"), "", "#f59e0b"),
        ("💵 سلف معتمدة", f"{adv_ok:,.0f}", "ريال", KIND_COLOR["advance"]),
        ("🌴 إجازات معتمدة", lv_ok, "يوم", KIND_COLOR["leave"]),
    ])

    admin_section("🗓 السجل الكامل (الأحدث أولًا)")
    if not mine:
        st.markdown('<div class="rq-empty">مفيش طلبات بالفلتر ده.</div>', unsafe_allow_html=True)
        return
    for r in mine:
        st.markdown(request_card(r, AR_ADMIN["kind"][r["kind"]], AR_ADMIN["status"][r["status"]],
                                 AR_ADMIN["labels"], admin=False, rtl=True), unsafe_allow_html=True)
    st.download_button("📥 تحميل سجل المندوب Excel", requests_excel(mine, perf),
                       file_name="rider_" + rid + "_requests.xlsx", key="rh_dl",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ------------------------------------------------------------
# Rider page
# ------------------------------------------------------------
if page == "Rider Performance":
    lang_name = st.radio("Language", list(LANGS.keys()), horizontal=True,
                         key="lang", label_visibility="collapsed")
    lang = LANGS[lang_name]
    T = TEXTS[lang]

    if lang in RTL_LANGS:
        st.markdown("""
        <style>
            .brand, .profile, .metric-card, .section-title, .footer-note, .tips, .seg-change, .tg-card, .rq-hero, .rq-card, .rq-form-title, .rq-empty,
            [data-testid="stForm"] {direction: rtl; text-align: right;}
            .brand {text-align: center;}
            .footer-note, .tips-closing {text-align: center;}
        </style>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="brand">
        <div class="brand-icon">🏆</div>
        <div class="brand-title">{html.escape(T["title"])}</div>
        <div class="brand-subtitle">{html.escape(T["subtitle"])}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("rider_lookup"):
        rider_id = st.text_input(T["id_label"], placeholder=T["id_ph"]).strip()
        search = st.form_submit_button(T["btn"], use_container_width=True, type="primary")

    if search:
        rider_id = re.sub(r"\.0$", "", rider_id)
        if not rider_id:
            st.session_state["last_id"] = None
            st.warning(T["need_id"])
        else:
            st.session_state["last_id"] = rider_id
            st.session_state["rider_view"] = "perf"

    shown_id = st.session_state.get("last_id")

    if shown_id:
        d = get_rider(shown_id)
        if d is None:
            st.error(T["not_found"])
        else:
            name = (d.get("rider_name") or "").strip() or T["rider"]
            seg = str(d.get("segment") or "").strip()
            if seg.lower() in ("", "nan", "none"):
                seg = "—"

            st.markdown(f"""
            <div class="profile">
                <div class="profile-name">{html.escape(name)}</div>
                <div class="profile-id">{html.escape(T["id_label"])}: {html.escape(str(d["rider_id"]))}</div>
                <div style="margin-top:14px;color:#cbd5e1;font-size:13px;">{html.escape(T["segment"])}</div>
                <div style="display:inline-block;min-width:76px;text-align:center;border-radius:14px;padding:10px 18px;font-size:25px;font-weight:900;margin-top:4px;background:{segment_color(seg)};color:#ffffff;">{html.escape(seg)}</div>
            </div>
            """, unsafe_allow_html=True)

            RQ = REQ[lang]
            view = st.session_state.get("rider_view", "perf")
            rq_n = rider_request_count(d["rider_id"])
            vb1, vb2 = st.columns(2)
            if vb1.button(RQ["tab_perf"], key="vw_perf", use_container_width=True,
                          type="primary" if view == "perf" else "secondary"):
                st.session_state["rider_view"] = "perf"
                st.rerun()
            if vb2.button(RQ["tab_req"] + (" (" + str(rq_n) + ")" if rq_n else ""), key="vw_req",
                          use_container_width=True, type="primary" if view == "req" else "secondary"):
                st.session_state["rider_view"] = "req"
                st.rerun()
            if view == "req":
                render_requests_page(d, lang)
                st.stop()

            render_segment_change(d, seg, T, CHANGE[lang])
            render_target_status(d, TARGETMSG[lang])

            st.markdown(f'<div class="section-title">{html.escape(T["overview"])}</div>',
                        unsafe_allow_html=True)

            cards = [
                (T["orders"], fmt_int(d["orders"]), None),
                (T["in_time"], fmt_int(d["orders_in_time"]), None),
                (T["late"], fmt_int(d["late_orders"]), None),
                (T["acceptance"], fmt_pct(d["acceptance_rate"]), None),
                (T["on_time"], fmt_pct(d["on_time_rate"]), None),
                (T["verification"], fmt_pct(d["verification_rate"]), None),
                (T["fail"], fmt_pct(d["fail_rate"]), None),
                (T["final"], fmt_pct(d["final_score"]), None),
                (T["segment"], seg, segment_color(seg)),
            ]
            cols = st.columns(3)
            for i, (label, value, color) in enumerate(cards):
                style = f' style="color:{color};"' if color else ""
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{html.escape(label)}</div>
                        <div class="metric-value"{style}>{html.escape(value)}</div>
                    </div>
                    """, unsafe_allow_html=True)

            render_tips(T, TIPS[lang])

            st.markdown(
                f'<div class="footer-note">{html.escape(T["footer"])}</div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------
# Admin page
# ------------------------------------------------------------
else:
    if not admin_login():
        st.stop()

    st.markdown("<style>.block-container {max-width: 1200px;}</style>", unsafe_allow_html=True)

    all_rows = list_requests(limit=100000)
    perf = perf_map()
    _pend = sum(1 for r in all_rows if r["status"] == "pending")
    _today = sum(1 for r in all_rows if str(r["created_at"])[:10] == today_ksa().isoformat())
    admin_banner(_pend, _today)

    F = admin_filters(perf)
    rows = apply_filters(all_rows, perf, F)

    tab_req, tab_dash, tab_rider, tab_upload, tab_data = st.tabs([
        "📨 الطلبات" + (" (" + str(_pend) + ")" if _pend else ""),
        "📈 لوحة المتابعة",
        "👤 سجل مندوب",
        "📥 رفع الشيت",
        "📊 البيانات الحالية",
    ])

    with tab_req:
        admin_requests_ui(all_rows, rows, perf)

    with tab_dash:
        admin_dashboard_ui(rows, perf)

    with tab_rider:
        admin_rider_history_ui(all_rows, perf)

    with tab_upload:
        st.info("⚠️ أي رفع جديد هيمسح كل البيانات القديمة ويحط الملف الجديد مكانها.")

        uploaded = st.file_uploader("Excel / CSV file", type=["xlsx", "xls", "csv"])

        if uploaded:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    raw = pd.read_csv(uploaded)
                else:
                    raw = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                st.stop()

            st.success(f"File loaded — {len(raw):,} rows, {len(raw.columns)} columns.")

            mapping = auto_map(raw.columns)
            missing = [FIELDS[k][0] for k, v in mapping.items() if v is None]

            if missing:
                st.error("الملف ناقصه أعمدة: " + "، ".join(missing))
                st.caption("الأعمدة المطلوبة: " + " | ".join(
                    ["rider_id", "Name", "chat with customer", "acceptance rate",
                     "verification_success_rate", "completed_orders", "completed_orders_in_time",
                     "Orders late", "failed_orders_by_rider", "on_time_delivery_score",
                     "fail_rate_score", "final_delivery_quality_score", "segment"]))
                st.stop()

            clean = build_clean_df(raw, mapping)

            if True:
                seg_counts = clean["segment"].value_counts().to_dict()
                st.caption("Segment في الملف: " + " | ".join(f"{k}: {v}" for k, v in seg_counts.items()))
                st.dataframe(clean.drop(columns=["uploaded_at"]).head(20),
                             use_container_width=True, hide_index=True)

                st.markdown("##### 🎯 إعدادات التارجت")
                g1, g2 = st.columns(2)
                tgt = g1.number_input("التارجت اليومي (طلب/يوم)", min_value=1, value=17, step=1)
                days = g2.number_input("عدد الأيام اللي الشيت مغطيها (من أول الشهر)", min_value=1,
                                       max_value=31, value=max(datetime.now().day - 1, 1), step=1)
                st.caption(f"المطلوب لحد دلوقتي من كل مندوب = {int(tgt)} × {int(days)} = {int(tgt) * int(days)} طلب")
                clean["target_per_day"] = int(tgt)
                clean["days_covered"] = int(days)

                if st.button("💾 مسح القديم ورفع الجديد", type="primary", use_container_width=True):
                    with_prev = replace_all_data(clean)
                    st.success(
                        f"✅ تم. البيانات القديمة اتمسحت واتحمّل {len(clean):,} رايدر "
                        f"({with_prev:,} منهم ليهم سيجمنت سابق)."
                    )

    with tab_data:
        current = get_all()
        if current.empty:
            st.info("No data uploaded yet.")
        else:
            st.metric("Riders", f"{len(current):,}")
            st.caption(f"آخر رفع: {current['uploaded_at'].iloc[0]}")

            r1, r2, r3 = st.columns([3, 2, 2])
            q_r = r1.text_input("🔎 بحث بالـ ID أو الاسم", key="cd_q").strip().lower()
            seg_opts = sorted(current["segment"].astype(str).unique().tolist())
            seg_r = r2.multiselect("السيجمنت", seg_opts, key="cd_seg", placeholder="الكل")
            below_r = r3.checkbox("الأقل من التارجت فقط", key="cd_below")

            view_df = current.copy()
            if q_r:
                view_df = view_df[view_df["rider_id"].astype(str).str.lower().str.contains(q_r, regex=False)
                                  | view_df["rider_name"].astype(str).str.lower().str.contains(q_r, regex=False)]
            if seg_r:
                view_df = view_df[view_df["segment"].astype(str).isin(seg_r)]
            if below_r and "target_per_day" in view_df.columns:
                view_df = view_df[(view_df["target_per_day"].fillna(0) * view_df["days_covered"].fillna(0))
                                  > view_df["orders"].fillna(0)]
            view_df = view_df.drop(columns=["uploaded_at"])

            st.caption(f"النتيجة: {len(view_df):,} مندوب")
            st.dataframe(view_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download Excel",
                data=to_excel_bytes({"المناديب": view_df}),
                file_name="rider_performance_" + today_ksa().isoformat() + ".xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
