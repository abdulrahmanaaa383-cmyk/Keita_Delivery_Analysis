import hmac
import html
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# RIDER PERFORMANCE PORTAL (HungerStation + Keeta)
# - المندوب يختار المشروع الأول (هنجر ستيشن / كيتا) وبعدين يدخل الـ ID
# - كل رفع جديد بيمسح بيانات المشروع ده بس (المشروع التاني مش بيتأثر)
# - الطلبات: إجازات فقط
# ============================================================

st.set_page_config(page_title="Rider Performance", page_icon="🏆", layout="centered")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "rider_performance.db"
TABLE = "rider_stats"          # هنجر ستيشن (زي ما كان)
KEETA_TABLE = "keeta_stats"    # كيتا

PROJECTS = {
    "keeta": {"name": "Keeta", "table": KEETA_TABLE, "logo": BASE_DIR / "keeta_logo.png"},
    "hunger": {"name": "HungerStation", "table": TABLE, "logo": BASE_DIR / "hunger_logo.png"},
}

# ---------------- HungerStation fields ----------------
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

# ---------------- Keeta fields (التيمبلت: id, name, accepted, Delivered Tasks, cancel,
# Rejected Tasks, Cancellation Rate from Delivery Issues, Experience_On-time Rate (D),
# orders late, valid day, level) ----------------
KEETA_FIELDS = {
    "rider_id":    ("id", True, ["id", "riderid", "courierid", "driverid"]),
    "rider_name":  ("name", False, ["name", "ridername", "couriername", "drivername"]),
    "accepted":    ("accepted", False, ["accepted", "acceptedtasks"]),
    "delivered":   ("Delivered Tasks", False, ["deliveredtasks", "delivered", "completedorders"]),
    "cancel":      ("cancel", False, ["cancel", "cancelled", "canceled"]),
    "rejected":    ("Rejected Tasks", False, ["rejectedtasks", "rejected"]),
    "cancel_rate": ("Cancellation Rate from Delivery Issues", False,
                    ["cancellationratefromdeliveryissues", "cancellationrate"]),
    "on_time_rate": ("Experience_On-time Rate (D)", False,
                     ["experienceontimeratd", "experienceontimerate", "ontimerate", "ontime"]),
    "orders_late": ("orders late", False, ["orderslate", "lateorders", "late"]),
    "valid_day":   ("valid day", False, ["validday", "validdays"]),
    "level":       ("level", False, ["level", "segment"]),
}
KEETA_INT = ["accepted", "delivered", "cancel", "rejected", "orders_late"]
KEETA_PCT = ["cancel_rate", "on_time_rate"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def norm(text):
    # يشيل أي مسافات / رموز / أقواس عشان "Experience_On-time Rate (D)" يتعرف
    return re.sub(r"[\W_]+", "", str(text).strip().lower())


def clean_id(series):
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def clean_text(series):
    s = series.fillna("").astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace({"nan": "", "None": "", "NaN": ""})


def to_pct_column(series):
    # لو القيم نصوص (زي "57.14%" أو "0.9981") نحولها لأرقام
    if not pd.api.types.is_numeric_dtype(series):
        txt = series.fillna("").astype(str)
        if txt.str.contains("%", regex=False).any():
            return pd.to_numeric(
                txt.str.replace("%", "", regex=False).str.strip(), errors="coerce"
            ).fillna(0.0)
        s = pd.to_numeric(txt.str.strip(), errors="coerce").fillna(0.0)
    else:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # لو العمود كله بين 0 و 1 يبقى نسبة -> نحولها لـ 0-100
    if len(s) and s.max() <= 1.0:
        s = s * 100
    return s


def to_int_column(series):
    txt = series.fillna("").astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(txt, errors="coerce").fillna(0).astype(int)


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


def fmt_text(v):
    v = "" if v is None else str(v).strip()
    return v if v and v.lower() not in ("nan", "none") else "—"


def segment_color(seg):
    return {
        "A": "#16a34a", "B": "#2563eb", "C": "#f59e0b",
        "D": "#f97316", "E": "#ef4444", "F": "#7f1d1d",
    }.get(str(seg).strip().upper(), "#64748b")


def to_float_safe(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


# ------------------------------------------------------------
# Database
# ------------------------------------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def replace_all_data(df):
    """هنجر: يمسح القديم ويحط الجديد، مع الاحتفاظ بآخر سيجمنت/مؤشرات لكل رايدر فقط."""
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


def replace_keeta_data(df):
    """كيتا: يمسح القديم ويحط الجديد."""
    conn = get_conn()
    df.to_sql(KEETA_TABLE, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def table_exists(table):
    conn = get_conn()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    conn.close()
    return row is not None


def get_rider(rider_id, project):
    table = PROJECTS[project]["table"]
    if not table_exists(table):
        return None
    conn = get_conn()
    df = pd.read_sql_query(
        f"SELECT * FROM {table} WHERE rider_id = ? LIMIT 1", conn, params=[rider_id]
    )
    conn.close()
    return None if df.empty else df.iloc[0].to_dict()


def get_all(project):
    table = PROJECTS[project]["table"]
    if not table_exists(table):
        return pd.DataFrame()
    order = "segment, rider_id" if project == "hunger" else "rider_id"
    conn = get_conn()
    df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order}", conn)
    conn.close()
    return df


# ------------------------------------------------------------
# Upload logic
# ------------------------------------------------------------
def auto_map(columns, fields=None):
    """يرجّع {field_key: اسم العمود في الإكسيل أو None}"""
    fields = fields or FIELDS
    by_norm = {norm(c): c for c in columns}
    mapping = {}
    for key, (_, _, aliases) in fields.items():
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


def build_keeta_df(raw, mapping):
    out = pd.DataFrame()
    out["rider_id"] = clean_id(raw[mapping["rider_id"]])
    out["rider_name"] = clean_text(raw[mapping["rider_name"]]) if mapping["rider_name"] else ""

    for c in KEETA_INT:
        out[c] = to_int_column(raw[mapping[c]]) if mapping[c] else 0
    for c in KEETA_PCT:
        out[c] = to_pct_column(raw[mapping[c]]) if mapping[c] else 0.0

    # valid day و level ممكن يكونوا فاضيين (بيظهروا آخر الشهر)
    out["valid_day"] = clean_text(raw[mapping["valid_day"]]) if mapping["valid_day"] else ""
    out["level"] = clean_text(raw[mapping["level"]]) if mapping["level"] else ""

    out = out[(out["rider_id"] != "") & (out["rider_id"].str.lower() != "nan")]
    out = out.drop_duplicates(subset="rider_id", keep="last").reset_index(drop=True)
    out["uploaded_at"] = datetime.now().isoformat(timespec="seconds")
    return out


# ------------------------------------------------------------
# Admin auth
# ------------------------------------------------------------
def admin_password():
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
    [data-testid="stImage"] {display:flex; justify-content:center;}
    [data-testid="stImage"] img {border-radius:22px;}
    .brand {text-align:center; margin-bottom:1.5rem;}
    .brand-icon {font-size:46px; line-height:1;}
    .brand-title {font-size:32px; font-weight:800; letter-spacing:-.7px;}
    .brand-subtitle {color:#6b7280; font-size:15px; margin-top:5px;}
    .proj-fallback {text-align:center; font-size:26px; font-weight:900; padding:40px 10px; border-radius:22px;
                    background:#f1f5f9; margin-bottom:8px;}
    .profile {background:linear-gradient(145deg,#0f172a,#1e293b); color:#fff;
              border-radius:24px; padding:30px; margin-top:25px;
              box-shadow:0 15px 45px rgba(15,23,42,.18);}
    .profile-name {font-size:30px; font-weight:800;}
    .profile-id {color:#cbd5e1; margin-top:4px; font-size:14px;}
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
    .rq-meta {margin-top:8px; font-size:11px; color:#94a3b8; font-weight:600;}
    .rq-empty {text-align:center; color:#94a3b8; padding:22px; background:#f8fafc; border:1px dashed #cbd5e1; border-radius:16px;}
    .rq-admin {direction:rtl; text-align:right;}
    .rq-kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:18px; direction:rtl;}
    .rq-kpi {background:#fff; border:1px solid #eceff7; border-radius:18px; padding:14px 18px; border-right:6px solid var(--c);
             box-shadow:0 6px 18px rgba(15,23,42,.05); text-align:right;}
    .rq-kpi-l {color:#64748b; font-weight:700; font-size:13px;} .rq-kpi-v {font-size:34px; font-weight:800; color:#0f172a; line-height:1.15;}
    .rq-kpi-s {color:#94a3b8; font-size:12px; font-weight:600;}

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
# Public page texts
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

# شاشة اختيار المشروع
CHOOSE = {
    "en": {"title": "Choose your project", "sub": "Select the project you work on to continue",
           "change": "↩ Change project"},
    "ar": {"title": "اختر المشروع", "sub": "اختر المشروع اللي بتشتغل عليه عشان تكمل",
           "change": "↩ تغيير المشروع"},
    "ur": {"title": "اپنا پروجیکٹ منتخب کریں", "sub": "جاری رکھنے کے لیے وہ پروجیکٹ منتخب کریں جس پر آپ کام کرتے ہیں",
           "change": "↩ پروجیکٹ تبدیل کریں"},
    "bn": {"title": "আপনার প্রজেক্ট বেছে নিন", "sub": "চালিয়ে যেতে আপনি যে প্রজেক্টে কাজ করেন সেটি নির্বাচন করুন",
           "change": "↩ প্রজেক্ট পরিবর্তন করুন"},
}

# نصوص كيتا
KTXT = {
    "en": {"accepted": "Accepted Tasks", "delivered": "Delivered Tasks", "cancel": "Cancelled",
           "rejected": "Rejected Tasks", "cancel_rate": "Cancellation Rate (Delivery Issues)",
           "on_time": "On-Time Rate", "late": "Late Orders", "valid_days": "Valid Days", "level": "Level",
           "month_end": "Valid Days and Level are published near the end of the month.",
           "footer": "Performance is based on the latest uploaded Keeta report."},
    "ar": {"accepted": "المهام المقبولة", "delivered": "المهام الموصلة", "cancel": "الملغية",
           "rejected": "المهام المرفوضة", "cancel_rate": "نسبة الإلغاء (مشاكل التوصيل)",
           "on_time": "نسبة التوصيل في الوقت", "late": "الطلبات المتأخرة", "valid_days": "الأيام الصالحة",
           "level": "المستوى",
           "month_end": "الأيام الصالحة والمستوى بيظهروا قرب نهاية الشهر.",
           "footer": "الأداء مبني على آخر تقرير كيتا تم رفعه."},
    "ur": {"accepted": "قبول شدہ ٹاسکس", "delivered": "ڈیلیور شدہ ٹاسکس", "cancel": "منسوخ",
           "rejected": "مسترد ٹاسکس", "cancel_rate": "منسوخی کی شرح (ڈیلیوری مسائل)",
           "on_time": "بروقت ڈیلیوری کی شرح", "late": "تاخیر سے آرڈرز", "valid_days": "درست دن",
           "level": "لیول",
           "month_end": "درست دن اور لیول مہینے کے آخر کے قریب ظاہر ہوتے ہیں۔",
           "footer": "کارکردگی کیٹا کی آخری اپ لوڈ کی گئی رپورٹ پر مبنی ہے۔"},
    "bn": {"accepted": "গৃহীত টাস্ক", "delivered": "ডেলিভারি হওয়া টাস্ক", "cancel": "বাতিল",
           "rejected": "প্রত্যাখ্যাত টাস্ক", "cancel_rate": "বাতিলের হার (ডেলিভারি সমস্যা)",
           "on_time": "সময়মতো ডেলিভারির হার", "late": "দেরিতে ডেলিভারি হওয়া অর্ডার", "valid_days": "বৈধ দিন",
           "level": "লেভেল",
           "month_end": "বৈধ দিন ও লেভেল মাসের শেষের দিকে প্রকাশ করা হয়।",
           "footer": "সর্বশেষ আপলোড করা কিটা রিপোর্টের ভিত্তিতে পারফরম্যান্স দেখানো হয়েছে।"},
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

TIPS_IMAGE = BASE_DIR / "tips_chat.jpg"


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


# ============================================================
# التارجت اليومي (هنجر ستيشن)
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
# 📨 نظام الطلبات: إجازات فقط
# المندوب يبعت من صفحته، والأدمن يشوفها ويوافق أو يرفض من تاب Requests
# ============================================================
KINDS = ["leave"]
KIND_COLOR = {"internet": "#0ea5e9", "advance": "#8b5cf6", "leave": "#14b8a6"}
STATUS_COLOR = {"pending": "#f59e0b", "approved": "#16a34a", "rejected": "#dc2626"}
KSA_TZ = timezone(timedelta(hours=3))

REQ = {
    "en": {
        "tab_perf": "📊 My Performance", "tab_req": "📨 My Requests",
        "title": "Leave Requests", "sub": "Send a leave request to the management and track its status",
        "h_leave": "Leave request",
        "f_note": "Notes (optional)", "f_from": "From", "f_to": "To", "submit": "Send request",
        "ok": "✅ Your request was sent (No. {n}). The management will review it.",
        "err_dates": "The end date must be the same day or after the start date.",
        "dup": "You already have a pending request of this type.",
        "my_title": "Your requests", "none": "You have no requests yet.",
        "pending": "⏳ Pending", "approved": "✅ Approved", "rejected": "❌ Rejected",
        "admin_note": "Management note", "days": "days", "sar": "SAR",
        "names": {"internet": "Internet request", "advance": "Advance request", "leave": "Leave request"},
    },
    "ar": {
        "tab_perf": "📊 أدائي", "tab_req": "📨 طلباتي",
        "title": "طلبات الإجازة", "sub": "ابعت طلب الإجازة للإدارة وتابع حالته",
        "h_leave": "طلب إجازة",
        "f_note": "ملاحظات (اختياري)", "f_from": "من", "f_to": "إلى", "submit": "إرسال الطلب",
        "ok": "✅ تم إرسال طلبك (رقم {n}). الإدارة هتراجعه.",
        "err_dates": "تاريخ النهاية لازم يكون نفس يوم البداية أو بعده.",
        "dup": "عندك طلب من النوع ده قيد المراجعة بالفعل.",
        "my_title": "طلباتك", "none": "مفيش طلبات لسه.",
        "pending": "⏳ قيد المراجعة", "approved": "✅ تمت الموافقة", "rejected": "❌ مرفوض",
        "admin_note": "ملاحظة الإدارة", "days": "يوم", "sar": "ريال",
        "names": {"internet": "طلب إنترنت", "advance": "طلب سلفة", "leave": "طلب إجازة"},
    },
    "ur": {
        "tab_perf": "📊 میری کارکردگی", "tab_req": "📨 میری درخواستیں",
        "title": "چھٹی کی درخواستیں", "sub": "انتظامیہ کو چھٹی کی درخواست بھیجیں اور اس کی حالت دیکھیں",
        "h_leave": "چھٹی کی درخواست",
        "f_note": "نوٹس (اختیاری)", "f_from": "سے", "f_to": "تک", "submit": "درخواست بھیجیں",
        "ok": "✅ آپ کی درخواست بھیج دی گئی (نمبر {n})۔ انتظامیہ جائزہ لے گی۔",
        "err_dates": "آخری تاریخ شروع کے دن یا اس کے بعد ہونی چاہیے۔",
        "dup": "اس قسم کی ایک درخواست پہلے سے زیر جائزہ ہے۔",
        "my_title": "آپ کی درخواستیں", "none": "ابھی تک کوئی درخواست نہیں۔",
        "pending": "⏳ زیر جائزہ", "approved": "✅ منظور", "rejected": "❌ مسترد",
        "admin_note": "انتظامیہ کا نوٹ", "days": "دن", "sar": "ریال",
        "names": {"internet": "انٹرنیٹ کی درخواست", "advance": "ایڈوانس کی درخواست", "leave": "چھٹی کی درخواست"},
    },
    "bn": {
        "tab_perf": "📊 আমার পারফরম্যান্স", "tab_req": "📨 আমার অনুরোধ",
        "title": "ছুটির অনুরোধ", "sub": "ম্যানেজমেন্টকে ছুটির অনুরোধ পাঠান এবং অবস্থা দেখুন",
        "h_leave": "ছুটির অনুরোধ",
        "f_note": "নোট (ঐচ্ছিক)", "f_from": "শুরু", "f_to": "শেষ", "submit": "অনুরোধ পাঠান",
        "ok": "✅ আপনার অনুরোধ পাঠানো হয়েছে (নম্বর {n})। ম্যানেজমেন্ট পর্যালোচনা করবে।",
        "err_dates": "শেষ তারিখ শুরুর দিনে বা তার পরে হতে হবে।",
        "dup": "এই ধরনের একটি অনুরোধ ইতিমধ্যে পর্যালোচনাধীন আছে।",
        "my_title": "আপনার অনুরোধসমূহ", "none": "এখনো কোনো অনুরোধ নেই।",
        "pending": "⏳ পর্যালোচনাধীন", "approved": "✅ অনুমোদিত", "rejected": "❌ প্রত্যাখ্যাত",
        "admin_note": "ম্যানেজমেন্টের নোট", "days": "দিন", "sar": "রিয়াল",
        "names": {"internet": "ইন্টারনেটের অনুরোধ", "advance": "অগ্রিমের অনুরোধ", "leave": "ছুটির অনুরোধ"},
    },
}

# نصوص الأدمن (عربي)
AR_ADMIN = {
    "kind": {"internet": "🌐 طلب إنترنت", "advance": "💵 طلب سلفة", "leave": "🌴 طلب إجازة"},
    "status": {"pending": "⏳ قيد المراجعة", "approved": "✅ تمت الموافقة", "rejected": "❌ مرفوض"},
    "labels": {"sar": "ريال", "days": "يوم", "admin_note": "ملاحظة الإدارة"},
}


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
    # ترقية الجدول القديم: عمود المشروع
    cols = [r[1] for r in conn.execute("PRAGMA table_info(requests)").fetchall()]
    if "project" not in cols:
        conn.execute("ALTER TABLE requests ADD COLUMN project TEXT DEFAULT 'hunger'")
    conn.commit()
    conn.close()


def now_ksa():
    return datetime.now(KSA_TZ).strftime("%Y-%m-%d %H:%M")


def add_request(rider_id, rider_name, kind, project="hunger", amount=None, phone=None,
                date_from=None, date_to=None, note="", lang="en"):
    init_requests_db()
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO requests (rider_id, rider_name, kind, amount, phone, date_from, date_to, note,
                                 status, created_at, updated_at, lang, project)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
        (str(rider_id), rider_name, kind, amount, phone, date_from, date_to, note,
         now_ksa(), now_ksa(), lang, project))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_requests(rider_id=None, status=None, kind=None, project=None, limit=500):
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
    if project:
        q += " AND COALESCE(project, 'hunger') = ?"
        params.append(project)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    for r in rows:
        r["project"] = r.get("project") or "hunger"
    return rows


def update_request(req_id, status, admin_note):
    conn = get_conn()
    conn.execute("UPDATE requests SET status = ?, admin_note = ?, updated_at = ? WHERE id = ?",
                 (status, admin_note.strip(), now_ksa(), int(req_id)))
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


def has_pending(rider_id, kind, project):
    return bool(list_requests(rider_id=rider_id, status="pending", kind=kind, project=project, limit=1))


def detail_text(r, labels):
    if r["kind"] == "internet":
        return "📱 " + (r.get("phone") or "—")
    if r["kind"] == "advance":
        return "💰 " + f"{(r.get('amount') or 0):,.0f} " + labels["sar"]
    try:
        n = (date.fromisoformat(r["date_to"]) - date.fromisoformat(r["date_from"])).days + 1
    except Exception:
        n = 1
    return "📅 " + str(r["date_from"]) + " → " + str(r["date_to"]) + "  (" + str(n) + " " + labels["days"] + ")"


def request_card(r, kind_label, status_label, labels, admin=False):
    e = html.escape
    color = KIND_COLOR.get(r["kind"], "#64748b")
    bcolor = STATUS_COLOR.get(r["status"], "#64748b")
    who = ""
    if admin:
        proj_name = PROJECTS.get(r.get("project") or "hunger", PROJECTS["hunger"])["name"]
        who = ('<div class="rq-who"><b>' + e(r.get("rider_name") or "—") + '</b><span>ID ' + e(str(r["rider_id"]))
               + "</span><span>" + e(proj_name) + "</span></div>")
    note = ""
    if r.get("note"):
        note = '<div class="rq-note">📝 ' + e(r["note"]) + "</div>"
    anote = ""
    if r.get("admin_note"):
        anote = '<div class="rq-anote"><b>' + e(labels["admin_note"]) + ":</b> " + e(r["admin_note"]) + "</div>"
    cls = "rq-card rq-admin" if admin else "rq-card"
    return (
        '<div class="' + cls + '" style="--c:' + color + '">'
        '<div class="rq-head"><span class="rq-kind">' + e(kind_label) + '</span>'
        '<span class="rq-badge" style="--b:' + bcolor + '">' + e(status_label) + "</span></div>"
        + who + '<div class="rq-detail">' + e(detail_text(r, labels)) + "</div>" + note + anote +
        '<div class="rq-meta">#' + str(r["id"]) + " · " + e(r["created_at"]) + "</div></div>"
    )


def rider_request_count(rider_id, project):
    try:
        return len(list_requests(rider_id=str(rider_id), project=project, limit=200))
    except Exception:
        return 0


def render_requests_page(d, lang, project):
    R = REQ[lang]
    e = html.escape
    rid = str(d["rider_id"])
    name = (d.get("rider_name") or "").strip()
    kind = "leave"

    st.markdown('<div class="rq-hero"><div class="rq-hero-t">🌴 ' + e(R["title"]) + '</div><div class="rq-hero-s">'
                + e(R["sub"]) + "</div></div>", unsafe_allow_html=True)

    with st.form("req_form_leave", clear_on_submit=True):
        st.markdown('<div class="rq-form-title" style="--c:' + KIND_COLOR[kind] + '">' + e(R["h_leave"]) + "</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        d_from = c1.date_input(R["f_from"], value=date.today(), min_value=date.today(), key="rq_from")
        d_to = c2.date_input(R["f_to"], value=date.today(), min_value=date.today(), key="rq_to")
        note = st.text_area(R["f_note"], height=80, max_chars=300, key="rq_note")
        sent = st.form_submit_button(R["submit"], type="primary", use_container_width=True)

    if sent:
        if d_to < d_from:
            st.error(R["err_dates"])
        elif has_pending(rid, kind, project):
            st.warning(R["dup"])
        else:
            new_id = add_request(
                rid, name, kind, project=project,
                date_from=d_from.isoformat(), date_to=d_to.isoformat(),
                note=(note or "").strip(), lang=lang)
            st.success(R["ok"].format(n=new_id))

    st.markdown('<div class="section-title">' + e(R["my_title"]) + "</div>", unsafe_allow_html=True)
    mine = list_requests(rider_id=rid, project=project, limit=50)
    if not mine:
        st.markdown('<div class="rq-empty">' + e(R["none"]) + "</div>", unsafe_allow_html=True)
    labels = {"sar": R["sar"], "days": R["days"], "admin_note": R["admin_note"]}
    for r in mine:
        st.markdown(request_card(r, R["names"][r["kind"]], R[r["status"]], labels), unsafe_allow_html=True)


def admin_requests_ui():
    e = html.escape
    rows = list_requests(limit=1000)
    pend = [r for r in rows if r["status"] == "pending"]
    by_kind = lambda k: [r for r in pend if r["kind"] == k]
    by_proj = lambda p: [r for r in pend if r["project"] == p]

    def kpi(label, value, sub, color):
        return ('<div class="rq-kpi" style="--c:' + color + '"><div class="rq-kpi-l">' + e(label) + '</div><div class="rq-kpi-v">'
                + str(value) + '</div><div class="rq-kpi-s">' + e(sub) + "</div></div>")

    st.markdown(
        '<div class="rq-kpis">'
        + kpi("⏳ طلبات قيد المراجعة", len(pend), "كل المشاريع", "#f59e0b")
        + kpi("🌴 إجازات", len(by_kind("leave")), "قيد المراجعة", KIND_COLOR["leave"])
        + kpi("🍔 HungerStation", len(by_proj("hunger")), "قيد المراجعة", "#7c3f2b")
        + kpi("🟡 Keeta", len(by_proj("keeta")), "قيد المراجعة", "#eab308")
        + "</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 2, 2])
    status_opts = {"⏳ قيد المراجعة": "pending", "✅ تمت الموافقة": "approved", "❌ مرفوضة": "rejected", "الكل": None}
    status_pick = f1.selectbox("الحالة", list(status_opts.keys()), key="rqa_status")
    proj_opts = {"الكل": None, "HungerStation": "hunger", "Keeta": "keeta"}
    proj_pick = f2.selectbox("المشروع", list(proj_opts.keys()), key="rqa_proj")
    q = f3.text_input("بحث بالـ ID أو الاسم", key="rqa_q").strip().lower()

    shown = [r for r in rows
             if (status_opts[status_pick] is None or r["status"] == status_opts[status_pick])
             and (proj_opts[proj_pick] is None or r["project"] == proj_opts[proj_pick])
             and (not q or q in str(r["rider_id"]).lower() or q in (r.get("rider_name") or "").lower())]

    if not shown:
        st.markdown('<div class="rq-empty">مفيش طلبات بالفلتر ده 🎉</div>', unsafe_allow_html=True)

    for r in shown[:40]:
        st.markdown(request_card(r, AR_ADMIN["kind"][r["kind"]], AR_ADMIN["status"][r["status"]],
                                 AR_ADMIN["labels"], admin=True), unsafe_allow_html=True)
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
    if len(shown) > 40:
        st.caption("معروض أحدث 40 طلب من " + str(len(shown)) + " — استخدم الفلتر لتضييق النتائج.")

    if rows:
        out = pd.DataFrame(rows)[["id", "created_at", "project", "rider_id", "rider_name", "kind", "status",
                                  "date_from", "date_to", "note", "admin_note"]]
        st.download_button("📥 تحميل كل الطلبات CSV", out.to_csv(index=False).encode("utf-8-sig"),
                           file_name="requests.csv", mime="text/csv", key="rqa_dl")


# ============================================================
# Rider page building blocks
# ============================================================
def reset_rider_view():
    for k in ("last_id", "rider_view"):
        st.session_state.pop(k, None)


def render_project_chooser(lang):
    C = CHOOSE[lang]
    st.markdown(f"""
    <div class="brand">
        <div class="brand-icon">🏆</div>
        <div class="brand-title">{html.escape(C["title"])}</div>
        <div class="brand-subtitle">{html.escape(C["sub"])}</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for col, key in zip(cols, ["keeta", "hunger"]):
        P = PROJECTS[key]
        with col:
            if P["logo"].exists():
                st.image(str(P["logo"]), width=170)
            else:
                st.markdown(f'<div class="proj-fallback">{html.escape(P["name"])}</div>', unsafe_allow_html=True)
            if st.button(P["name"], key="pick_" + key, use_container_width=True, type="primary"):
                st.session_state["project"] = key
                reset_rider_view()
                st.rerun()


def render_profile(d, name, id_label, badge_label, badge_value, badge_color):
    st.markdown(f"""
    <div class="profile">
        <div class="profile-name">{html.escape(name)}</div>
        <div class="profile-id">{html.escape(id_label)}: {html.escape(str(d["rider_id"]))}</div>
        <div style="margin-top:14px;color:#cbd5e1;font-size:13px;">{html.escape(badge_label)}</div>
        <div style="display:inline-block;min-width:76px;text-align:center;border-radius:14px;padding:10px 18px;font-size:25px;font-weight:900;margin-top:4px;background:{badge_color};color:#ffffff;">{html.escape(badge_value)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_cards(cards):
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


def render_hunger_body(d, seg, T, lang):
    render_segment_change(d, seg, T, CHANGE[lang])
    render_target_status(d, TARGETMSG[lang])

    st.markdown(f'<div class="section-title">{html.escape(T["overview"])}</div>', unsafe_allow_html=True)
    render_cards([
        (T["orders"], fmt_int(d["orders"]), None),
        (T["in_time"], fmt_int(d["orders_in_time"]), None),
        (T["late"], fmt_int(d["late_orders"]), None),
        (T["acceptance"], fmt_pct(d["acceptance_rate"]), None),
        (T["on_time"], fmt_pct(d["on_time_rate"]), None),
        (T["verification"], fmt_pct(d["verification_rate"]), None),
        (T["fail"], fmt_pct(d["fail_rate"]), None),
        (T["final"], fmt_pct(d["final_score"]), None),
        (T["segment"], seg, segment_color(seg)),
    ])
    render_tips(T, TIPS[lang])
    st.markdown(f'<div class="footer-note">{html.escape(T["footer"])}</div>', unsafe_allow_html=True)


def render_keeta_body(d, T, K):
    st.markdown(f'<div class="section-title">{html.escape(T["overview"])}</div>', unsafe_allow_html=True)
    valid_day = fmt_text(d.get("valid_day"))
    level = fmt_text(d.get("level"))
    render_cards([
        (K["accepted"], fmt_int(d.get("accepted")), None),
        (K["delivered"], fmt_int(d.get("delivered")), None),
        (K["cancel"], fmt_int(d.get("cancel")), None),
        (K["rejected"], fmt_int(d.get("rejected")), None),
        (K["cancel_rate"], fmt_pct(d.get("cancel_rate")), None),
        (K["on_time"], fmt_pct(d.get("on_time_rate")), None),
        (K["late"], fmt_int(d.get("orders_late")), None),
        (K["valid_days"], valid_day, None),
        (K["level"], level, "#14b8a6" if level != "—" else None),
    ])
    if valid_day == "—" or level == "—":
        st.markdown(f'<div class="tips-note">ℹ️ {html.escape(K["month_end"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="footer-note">{html.escape(K["footer"])}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Public page
# ------------------------------------------------------------
if page == "Rider Performance":
    lang_name = st.radio("Language", list(LANGS.keys()), horizontal=True,
                         key="lang", label_visibility="collapsed")
    lang = LANGS[lang_name]
    T = TEXTS[lang]

    if lang in RTL_LANGS:
        st.markdown("""
        <style>
            .brand, .profile, .metric-card, .section-title, .footer-note, .tips, .tips-note, .seg-change, .tg-card, .rq-hero, .rq-card, .rq-form-title, .rq-empty,
            [data-testid="stForm"] {direction: rtl; text-align: right;}
            .brand {text-align: center;}
            .footer-note, .tips-closing {text-align: center;}
        </style>
        """, unsafe_allow_html=True)

    # ---- الخطوة 1: اختيار المشروع ----
    project = st.session_state.get("project")
    if project not in PROJECTS:
        render_project_chooser(lang)
        st.stop()

    P = PROJECTS[project]
    CH = CHOOSE[lang]

    tb1, tb2 = st.columns([1, 3])
    with tb1:
        if P["logo"].exists():
            st.image(str(P["logo"]), width=64)
    with tb2:
        if st.button(CH["change"], key="chg_proj"):
            st.session_state.pop("project", None)
            reset_rider_view()
            st.rerun()

    st.markdown(f"""
    <div class="brand">
        <div class="brand-icon">🏆</div>
        <div class="brand-title">{html.escape(T["title"])} — {html.escape(P["name"])}</div>
        <div class="brand-subtitle">{html.escape(T["subtitle"])}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- الخطوة 2: إدخال الـ ID ----
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
        d = get_rider(shown_id, project)
        if d is None:
            st.error(T["not_found"])
        else:
            name = (d.get("rider_name") or "").strip() or T["rider"]

            if project == "hunger":
                seg = str(d.get("segment") or "").strip()
                if seg.lower() in ("", "nan", "none"):
                    seg = "—"
                render_profile(d, name, T["id_label"], T["segment"], seg, segment_color(seg))
            else:
                K = KTXT[lang]
                level = fmt_text(d.get("level"))
                render_profile(d, name, T["id_label"], K["level"], level,
                               "#14b8a6" if level != "—" else "#64748b")

            # تابات: الأداء / الطلبات (إجازات)
            RQ = REQ[lang]
            view = st.session_state.get("rider_view", "perf")
            rq_n = rider_request_count(d["rider_id"], project)
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
                render_requests_page(d, lang, project)
                st.stop()

            if project == "hunger":
                render_hunger_body(d, seg, T, lang)
            else:
                render_keeta_body(d, T, KTXT[lang])

# ------------------------------------------------------------
# Admin page
# ------------------------------------------------------------
def admin_upload_hunger():
    st.info("⚠️ أي رفع جديد لهنجر ستيشن هيمسح بيانات هنجر القديمة (كيتا مش هتتأثر).")

    uploaded = st.file_uploader("Excel / CSV file", type=["xlsx", "xls", "csv"], key="up_hunger")
    if not uploaded:
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    st.success(f"File loaded — {len(raw):,} rows, {len(raw.columns)} columns.")

    mapping = auto_map(raw.columns, FIELDS)
    missing = [FIELDS[k][0] for k, v in mapping.items() if v is None]

    if missing:
        st.error("الملف ناقصه أعمدة: " + "، ".join(missing))
        st.caption("الأعمدة المطلوبة: " + " | ".join(
            ["rider_id", "Name", "chat with customer", "acceptance rate",
             "verification_success_rate", "completed_orders", "completed_orders_in_time",
             "Orders late", "failed_orders_by_rider", "on_time_delivery_score",
             "fail_rate_score", "final_delivery_quality_score", "segment"]))
        return

    clean = build_clean_df(raw, mapping)

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

    if st.button("💾 مسح القديم ورفع الجديد", type="primary", use_container_width=True, key="save_hunger"):
        with_prev = replace_all_data(clean)
        st.success(
            f"✅ تم. البيانات القديمة اتمسحت واتحمّل {len(clean):,} رايدر "
            f"({with_prev:,} منهم ليهم سيجمنت سابق)."
        )


def admin_upload_keeta():
    st.info("⚠️ أي رفع جديد لكيتا هيمسح بيانات كيتا القديمة (هنجر ستيشن مش هتتأثر).")
    st.caption("التيمبلت: id | name | accepted | Delivered Tasks | cancel | Rejected Tasks | "
               "Cancellation Rate from Delivery Issues | Experience_On-time Rate (D) | orders late | valid day | level "
               "— الـ valid day والـ level ممكن يكونوا فاضيين لحد آخر الشهر.")

    uploaded = st.file_uploader("Keeta Excel / CSV file", type=["xlsx", "xls", "csv"], key="up_keeta")
    if not uploaded:
        return

    try:
        # dtype=str عشان الـ ID الطويل (16 رقم) ميتبوظش
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded, dtype=str)
        else:
            raw = pd.read_excel(uploaded, dtype=str)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    st.success(f"File loaded — {len(raw):,} rows, {len(raw.columns)} columns.")

    mapping = auto_map(raw.columns, KEETA_FIELDS)
    if mapping["rider_id"] is None:
        st.error("الملف مفيهوش عمود id.")
        return

    missing = [KEETA_FIELDS[k][0] for k, v in mapping.items() if v is None]
    if missing:
        st.warning("أعمدة مش موجودة في الملف (هتظهر 0 / —): " + "، ".join(missing))

    clean = build_keeta_df(raw, mapping)

    no_valid = int((clean["valid_day"] == "").sum())
    no_level = int((clean["level"] == "").sum())
    st.caption(f"الرايدرز: {len(clean):,} | من غير valid day: {no_valid:,} | من غير level: {no_level:,}")
    st.dataframe(clean.drop(columns=["uploaded_at"]).head(20),
                 use_container_width=True, hide_index=True)

    if st.button("💾 مسح القديم ورفع الجديد", type="primary", use_container_width=True, key="save_keeta"):
        replace_keeta_data(clean)
        st.success(f"✅ تم. بيانات كيتا القديمة اتمسحت واتحمّل {len(clean):,} رايدر.")


if page != "Rider Performance":
    if not admin_login():
        st.stop()

    st.title("⚙️ Performance Admin")
    _pend = count_requests("pending")
    tab_upload, tab_data, tab_req = st.tabs(["📥 Upload", "📊 Current Data",
                                             "📨 Requests" + (" (" + str(_pend) + ")" if _pend else "")])

    with tab_upload:
        up_proj = st.radio("المشروع", ["Keeta", "HungerStation"], horizontal=True, key="up_proj")
        if up_proj == "Keeta":
            admin_upload_keeta()
        else:
            admin_upload_hunger()

    with tab_data:
        dp = st.radio("المشروع", ["Keeta", "HungerStation"], horizontal=True, key="data_proj")
        dp_key = "keeta" if dp == "Keeta" else "hunger"
        current = get_all(dp_key)
        if current.empty:
            st.info("No data uploaded yet.")
        else:
            st.metric("Riders", f"{len(current):,}")
            st.caption(f"آخر رفع: {current['uploaded_at'].iloc[0]}")
            st.dataframe(current.drop(columns=["uploaded_at"]),
                         use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download Current Data",
                data=current.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{dp_key}_performance_current.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_req:
        admin_requests_ui()
