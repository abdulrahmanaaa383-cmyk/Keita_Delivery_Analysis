import copy
import hmac
import html
import json
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# RIDER PERFORMANCE PORTAL (HungerStation + Keeta)
# - المندوب يختار المشروع الأول (هنجر ستيشن / كيتا) وبعدين يدخل الـ ID
# - كل رفع جديد بيمسح بيانات المشروع ده بس (المشروع التاني مش بيتأثر)
# - الطلبات: إجازات فقط
# - محرك تعليمات ذكي: كل مندوب بيشوف تعليمات على حسب أرقامه (تلقائي) + رسائل يدوية من الأدمن
# - Admin Control Center: Dashboard / Riders / Coaching / Upload / Requests
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

# ---------------- Keeta fields ----------------
KEETA_FIELDS = {
    "rider_id":    ("id", True, ["id", "riderid", "courierid", "driverid", "captainid", "userid", "accountid",
                                 "keetaid", "courieridkeeta"]),
    "rider_name":  ("name", False, ["name", "ridername", "couriername", "drivername", "captainname"]),
    "accepted":    ("accepted", False, ["accepted", "acceptedtasks"]),
    "delivered":   ("Delivered Tasks", False, ["deliveredtasks", "delivered", "completedorders"]),
    "cancel":      ("cancel", False, ["cancel", "cancelled", "canceled"]),
    "rejected":    ("Rejected Tasks", False, ["rejectedtasks", "rejected"]),
    "cancel_rate": ("Cancellation Rate from Delivery Issues", False,
                    ["cancellationratefromdeliveryissues", "cancellationrate"]),
    "on_time_rate": ("Experience_On-time Rate (D)", False,
                     ["experienceontimerated", "experienceontimeratd", "experienceontimerate", "ontimerate", "ontime"]),
    "orders_late": ("orders late", False, ["orderslate", "lateorders", "late"]),
    "valid_day":   ("valid day", False, ["validday", "validdays"]),
    "level":       ("level", False, ["level", "segment"]),
}
KEETA_INT = ["accepted", "delivered", "cancel", "rejected", "orders_late"]
KEETA_PCT = ["cancel_rate", "on_time_rate"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm(text):
    # يشيل أي مسافات / رموز / أقواس عشان "Experience_On-time Rate (D)" يتعرف
    return re.sub(r"[\W_]+", "", str(text).strip().lower())


def _fix_id(v):
    v = str(v).strip().translate(AR_DIGITS)
    if re.fullmatch(r"\d+(\.\d+)?[eE]\+?\d+", v):   # لو الإكسيل حوّل الـ ID لصيغة علمية
        try:
            v = str(int(Decimal(v)))
        except Exception:
            pass
    return re.sub(r"\.0+$", "", v)


def normalize_id_text(v):
    return _fix_id(v)


def clean_id(series):
    s = series.fillna("").astype(str).map(_fix_id)
    return s.replace({"nan": "", "None": "", "NaN": ""})


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


def init_aux_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coach_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL, scope TEXT NOT NULL,
            rider_id TEXT, rule TEXT, title TEXT, body TEXT NOT NULL,
            active INTEGER DEFAULT 1, created_at TEXT
        )""")
    conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()


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


init_aux_db()


# ------------------------------------------------------------
# Upload logic (قراءة ذكية + مطابقة أعمدة)
# ------------------------------------------------------------
def read_table(uploaded, fields):
    """يقرأ Excel/CSV كنصوص، ويكتشف صف العناوين تلقائيًا (حتى لو مش أول صف)."""
    name = uploaded.name.lower()
    raw = None
    if name.endswith(".csv"):
        for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
            try:
                uploaded.seek(0)
                raw = pd.read_csv(uploaded, header=None, dtype=str, encoding=enc, keep_default_na=False)
                break
            except Exception:
                continue
        if raw is None:
            raise ValueError("تعذّر قراءة ترميز ملف الـ CSV")
    else:
        uploaded.seek(0)
        raw = pd.read_excel(uploaded, header=None, dtype=str)

    raw = raw.fillna("")
    aliases = {a for (_, _, al) in fields.values() for a in al}
    best, best_score = 0, -1
    for i in range(min(15, len(raw))):
        score = sum(1 for c in raw.iloc[i].tolist() if norm(c) in aliases)
        if score > best_score:
            best, best_score = i, score

    cols, seen = [], {}
    for j, c in enumerate(raw.iloc[best].tolist()):
        c = str(c).strip()
        if not c or c.lower() == "nan":
            c = f"Column {j + 1}"
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)

    df = raw.iloc[best + 1:].reset_index(drop=True)
    df.columns = cols
    blank = df.apply(lambda s: s.astype(str).str.strip()).eq("").all(axis=1)
    df = df.loc[~blank].reset_index(drop=True)
    return df, best


def auto_map(columns, fields=None):
    """يرجّع {field_key: اسم العمود في الإكسيل أو None}"""
    fields = fields or FIELDS
    by_norm = {}
    for c in columns:
        by_norm.setdefault(norm(c), c)
    mapping = {}
    for key, (_, _, aliases) in fields.items():
        hit = next((by_norm[a] for a in aliases if a in by_norm), None)
        if hit is None and key == "rider_id":
            # أي عمود اسمه بينتهي بـ id (Courier ID / Rider_ID ...) أو بيبدأ بـ id
            for n, c in by_norm.items():
                if (n.endswith("id") and n not in ("valid", "paid", "android", "void")) or \
                        (n.startswith("id") and len(n) <= 4):
                    hit = c
                    break
        mapping[key] = hit
    return mapping


def guess_id_column(raw):
    """لو مفيش عمود اسمه ID: نختار العمود اللي أغلب قيمه أرقام طويلة (6+ خانات)."""
    best, best_ratio = None, 0.0
    for c in raw.columns:
        vals = raw[c].astype(str).map(_fix_id)
        vals = vals[vals != ""]
        if len(vals) == 0:
            continue
        ratio = vals.str.fullmatch(r"\d{6,}").mean()
        if ratio > best_ratio:
            best, best_ratio = c, ratio
    return best if best_ratio >= 0.8 else None


def manual_map(raw, mapping, fields, prefix):
    """لو في أعمدة مش اتعرفت تلقائي، الأدمن يختارها يدويًا."""
    miss = [k for k, v in mapping.items() if v is None]
    if not miss:
        return mapping
    if mapping.get("rider_id") is None:
        g = guess_id_column(raw)
        if g is not None:
            mapping["rider_id"] = g
            st.info(f"تم اختيار عمود «{g}» كـ ID تلقائيًا لأن قيمه أرقام طويلة. لو غلط غيّره من تحت.")
            miss = [k for k, v in mapping.items() if v is None]
            if not miss:
                return mapping
    none_label = "— لا يوجد —"
    opts = [none_label] + list(raw.columns)
    with st.expander("🧩 مطابقة الأعمدة يدويًا (للأعمدة اللي ما اتعرفتش)", expanded=mapping["rider_id"] is None):
        for k in miss:
            label = fields[k][0] + (" *" if fields[k][1] else "")
            pick = st.selectbox(label, opts, key=f"{prefix}_map_{k}")
            if pick != none_label:
                mapping[k] = pick
    return mapping


def build_clean_df(raw, mapping):
    out = pd.DataFrame()
    out["rider_id"] = clean_id(raw[mapping["rider_id"]])

    if mapping["rider_name"]:
        out["rider_name"] = clean_text(raw[mapping["rider_name"]])
    else:
        out["rider_name"] = ""

    for c in INT_COLS:
        out[c] = to_int_column(raw[mapping[c]]) if mapping[c] else 0

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
    try:
        pw = st.secrets.get("ADMIN_PASSWORD")
        if pw:
            return str(pw)
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

    /* ---------- Smart coaching cards ---------- */
    .co-wrap {margin-top:8px;}
    .co-card {border-radius:18px; padding:16px 18px; margin-bottom:12px; border:1px solid; border-inline-start:6px solid;
              box-shadow:0 5px 18px rgba(15,23,42,.05);}
    .co-head {display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:6px;}
    .co-title {font-size:17px; font-weight:800;}
    .co-badge {border-radius:999px; padding:3px 11px; font-size:11px; font-weight:800; background:rgba(255,255,255,.75);}
    .co-body {font-size:14.5px; line-height:1.75;}
    .co-card ul {margin:4px 0 0; padding-inline-start:20px;}
    .co-card li {margin:3px 0;}
    .co-crit {background:#fef2f2; border-color:#fecaca; border-inline-start-color:#dc2626; color:#7f1d1d;}
    .co-warn {background:#fffbeb; border-color:#fde68a; border-inline-start-color:#f59e0b; color:#78350f;}
    .co-info {background:#eff6ff; border-color:#bfdbfe; border-inline-start-color:#2563eb; color:#1e3a8a;}
    .co-good {background:#f0fdf4; border-color:#bbf7d0; border-inline-start-color:#16a34a; color:#14532d;}
    .co-note {background:#faf5ff; border-color:#e9d5ff; border-inline-start-color:#9333ea; color:#581c87;}

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
# 🧠 محرك التعليمات الذكي (تلقائي + يدوي)
# ============================================================
DEFAULT_SETTINGS = {
    "keeta": {"late_pct": 5.0, "on_time_min": 95.0, "rejected_max": 3, "cancel_max": 2,
              "cancel_rate_max": 2.0, "low_orders": 60},
    "hunger": {"late_pct": 5.0, "on_time_min": 98.0, "verification_min": 100.0, "fail_min": 100.0,
               "acceptance_min": 90.0, "chat_min": 80.0},
    "disabled": {"keeta": [], "hunger": []},
}

SETTING_LABELS = {
    "keeta": {
        "late_pct": "نسبة التأخير القصوى % (متأخر ÷ موصّل)",
        "on_time_min": "أقل نسبة أون تايم %",
        "rejected_max": "عدد الرفض اللي يبدأ عنده التحذير",
        "cancel_max": "عدد الإلغاء اللي يبدأ عنده التحذير",
        "cancel_rate_max": "نسبة الإلغاء القصوى %",
        "low_orders": "الحد الأدنى للطلبات الموصلة",
    },
    "hunger": {
        "late_pct": "نسبة التأخير القصوى % (متأخر ÷ طلبات)",
        "on_time_min": "أقل نسبة أون تايم %",
        "verification_min": "أقل نسبة تحقق %",
        "fail_min": "أقل درجة Fail Order %",
        "acceptance_min": "أقل نسبة قبول %",
        "chat_min": "أقل نسبة تواصل مع العميل %",
    },
}

RULE_AR = {
    "late": "⏱️ تأخير", "reject": "⚠️ رفض متكرر", "accept": "⚠️ قبول ضعيف", "cancel": "🚫 إلغاء",
    "fail": "❌ طلبات فاشلة", "verify": "🔐 تحقق", "chat": "💬 تواصل", "low": "📉 طلبات قليلة",
}
ISSUE_KEYS = list(RULE_AR.keys())
PROJECT_RULES = {
    "keeta": ["late", "reject", "cancel", "low"],
    "hunger": ["late", "accept", "fail", "verify", "chat", "low"],
}
SEV_RANK = {"critical": 0, "warn": 1, "info": 2, "good": 3}


class _Safe(dict):
    def __missing__(self, key):
        return "—"


def load_settings():
    s = copy.deepcopy(DEFAULT_SETTINGS)
    try:
        conn = get_conn()
        row = conn.execute("SELECT value FROM app_settings WHERE key='coach'").fetchone()
        conn.close()
        if row:
            saved = json.loads(row[0])
            for p in ("keeta", "hunger"):
                s[p].update(saved.get(p, {}))
                s["disabled"][p] = saved.get("disabled", {}).get(p, [])
    except Exception:
        pass
    return s


def save_settings(s):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('coach', ?)", (json.dumps(s),))
    conn.commit()
    conn.close()


def _f(v, default=0.0):
    x = to_float_safe(v)
    return default if x is None else x


def evaluate(d, project, S):
    """يرجّع قايمة (rule_key, severity, ctx) مرتبة حسب الخطورة."""
    P = S[project]
    off = set(S["disabled"].get(project, []))
    items = []

    if project == "keeta":
        deliv = int(_f(d.get("delivered")))
        late = int(_f(d.get("orders_late")))
        rej = int(_f(d.get("rejected")))
        canc = int(_f(d.get("cancel")))
        crate = _f(d.get("cancel_rate"))
        ot = _f(d.get("on_time_rate"))
        late_pct = (late / deliv * 100) if deliv > 0 else 0.0
        ctx = {"late": late, "late_pct": f"{late_pct:.1f}", "on_time": f"{ot:.1f}", "rejected": rej,
               "cancel": canc, "cancel_rate": f"{crate:.1f}", "orders": deliv,
               "expected": int(P["low_orders"]), "missing": max(int(P["low_orders"]) - deliv, 0)}

        if deliv > 0 and (late_pct > P["late_pct"] or (0 < ot < P["on_time_min"])):
            crit = late_pct >= 2 * P["late_pct"] or (0 < ot < P["on_time_min"] - 5)
            items.append(("late", "critical" if crit else "warn", ctx))
        if P["rejected_max"] > 0 and rej >= P["rejected_max"]:
            items.append(("reject", "critical", ctx))
        if (P["cancel_max"] > 0 and canc >= P["cancel_max"]) or crate > P["cancel_rate_max"]:
            crit = canc >= 2 * P["cancel_max"] or crate >= 2 * P["cancel_rate_max"]
            items.append(("cancel", "critical" if crit else "warn", ctx))
        if P["low_orders"] > 0 and deliv < P["low_orders"]:
            items.append(("low", "warn", ctx))
    else:
        orders = int(_f(d.get("orders")))
        late = int(_f(d.get("late_orders")))
        failed = int(_f(d.get("failed_orders")))
        ot = _f(d.get("on_time_rate"))
        ver = _f(d.get("verification_rate"))
        frate = _f(d.get("fail_rate"))
        acc = _f(d.get("acceptance_rate"))
        chat = _f(d.get("chat_rate"))
        late_pct = (late / orders * 100) if orders > 0 else 0.0
        t = int(_f(d.get("target_per_day")))
        days = int(_f(d.get("days_covered")))
        expected = t * days
        ctx = {"late": late, "late_pct": f"{late_pct:.1f}", "on_time": f"{ot:.1f}", "failed": failed,
               "fail_rate": f"{frate:.1f}", "verification": f"{ver:.1f}", "acceptance": f"{acc:.1f}",
               "chat": f"{chat:.1f}", "orders": orders, "expected": expected,
               "missing": max(expected - orders, 0)}

        if orders > 0 and (late_pct > P["late_pct"] or (0 < ot < P["on_time_min"])):
            items.append(("late", "critical" if (0 < ot < P["on_time_min"] - 5) else "warn", ctx))
        if 0 < acc < P["acceptance_min"]:
            items.append(("accept", "critical" if acc < P["acceptance_min"] - 15 else "warn", ctx))
        if failed > 0 or 0 < frate < P["fail_min"]:
            crit = failed >= 3 or (0 < frate < P["fail_min"] - 5)
            items.append(("fail", "critical" if crit else "warn", ctx))
        if 0 < ver < P["verification_min"]:
            items.append(("verify", "warn", ctx))
        if 0 < chat < P["chat_min"]:
            items.append(("chat", "info", ctx))
        if expected > 0 and orders < expected:
            items.append(("low", "warn", ctx))

    items = [i for i in items if i[0] not in off]
    base_ctx = items[0][2] if items else {}
    if any(k in ("late", "reject", "accept", "cancel", "fail") for k, _, _ in items):
        items.append(("proof", "info", base_ctx))
    if not any(s in ("critical", "warn") for _, s, _ in items):
        items.append(("good", "good", base_ctx))
    items.sort(key=lambda x: SEV_RANK[x[1]])
    return items


COACH_UI = {
    "en": {"title": "📌 Your personal instructions", "mgmt": "Message from management",
           "critical": "Urgent", "warn": "Warning", "info": "Tip", "good": "Great"},
    "ar": {"title": "📌 تعليمات خاصة بك", "mgmt": "رسالة من الإدارة",
           "critical": "عاجل", "warn": "تنبيه", "info": "نصيحة", "good": "ممتاز"},
    "ur": {"title": "📌 آپ کے لیے خصوصی ہدایات", "mgmt": "انتظامیہ کا پیغام",
           "critical": "فوری", "warn": "انتباہ", "info": "مشورہ", "good": "بہترین"},
    "bn": {"title": "📌 আপনার জন্য বিশেষ নির্দেশনা", "mgmt": "ম্যানেজমেন্টের বার্তা",
           "critical": "জরুরি", "warn": "সতর্কতা", "info": "পরামর্শ", "good": "চমৎকার"},
}

COACH = {
    "late": {
        "ar": ("⏱️ انتبه لوقت التوصيل", [
            "نسبة تأخيرك {late_pct}% ({late} طلب متأخر) والتوصيل في الوقت {on_time}%.",
            "بعد استلام الطلب اتحرك فورًا ولا تتأخر في المطعم، وإذا الطلب متأخر بلّغ العميل والدعم.",
            "أرسل للعميل رسالة في الشات: عند المطعم، وبعد الاستلام، وعند الوصول له.",
            "اختر أقصر طريق وراقب المسافة، وإذا وجدت زحمة صوّرها وأرسلها للعميل.",
        ]),
        "en": ("⏱️ Watch your delivery time", [
            "Your late rate is {late_pct}% ({late} late orders) and on-time is {on_time}%.",
            "Leave the restaurant right after pickup; if the order is delayed, tell the customer and support.",
            "Message the customer in chat: at the restaurant, after pickup, and on arrival.",
            "Take the shortest route, track the distance, and photograph any traffic and send it to the customer.",
        ]),
        "ur": ("⏱️ ڈیلیوری کے وقت کا خیال رکھیں", [
            "آپ کی تاخیر کی شرح {late_pct}% ({late} لیٹ آرڈرز) اور بروقت ڈیلیوری {on_time}% ہے۔",
            "آرڈر لینے کے فوراً بعد نکلیں؛ تاخیر ہو تو کسٹمر اور سپورٹ کو بتائیں۔",
            "ریسٹورنٹ پہنچ کر، آرڈر لینے کے بعد اور کسٹمر کے پاس پہنچ کر چیٹ میں پیغام بھیجیں۔",
            "مختصر راستہ چنیں، اور ٹریفک ہو تو تصویر لے کر کسٹمر کو بھیجیں۔",
        ]),
        "bn": ("⏱️ ডেলিভারির সময়ের দিকে খেয়াল রাখুন", [
            "আপনার দেরির হার {late_pct}% ({late}টি দেরি অর্ডার) এবং সময়মতো ডেলিভারি {on_time}%।",
            "অর্ডার নেওয়ার পরপরই রওনা দিন; দেরি হলে কাস্টমার ও সাপোর্টকে জানান।",
            "রেস্টুরেন্টে পৌঁছে, অর্ডার নেওয়ার পর এবং কাস্টমারের কাছে পৌঁছে চ্যাটে মেসেজ দিন।",
            "সবচেয়ে ছোট রাস্তা বেছে নিন, যানজট হলে ছবি তুলে কাস্টমারকে পাঠান।",
        ]),
    },
    "reject": {
        "ar": ("⚠️ الرفض المتكرر يعرّض حسابك للإيقاف", [
            "رفضت {rejected} مهمة.",
            "الرفض المتكرر قد يعرّض حسابك للإيقاف، وقد يتم الخصم عليك مبالغ مالية.",
            "اقبل فقط الطلبات التي تستطيع تنفيذها، وإذا كان عندك ظرف بلّغ الدعم بدل الرفض.",
            "الطلبات المقبولة والمنفذة بسرعة ترفع مستواك وتزيد الطلبات اللي تجيك.",
        ]),
        "en": ("⚠️ Repeated rejections put your account at risk", [
            "You rejected {rejected} tasks.",
            "Repeated rejections can get your account suspended and money may be deducted from you.",
            "Only accept orders you can complete; if you have a problem, contact support instead of rejecting.",
            "Accepted and fast-delivered orders raise your level and bring you more orders.",
        ]),
        "ur": ("⚠️ بار بار مسترد کرنا آپ کے اکاؤنٹ کو خطرے میں ڈالتا ہے", [
            "آپ نے {rejected} ٹاسکس مسترد کیے۔",
            "بار بار مسترد کرنے سے اکاؤنٹ معطل ہو سکتا ہے اور رقم کٹ سکتی ہے۔",
            "صرف وہی آرڈر قبول کریں جو آپ مکمل کر سکیں؛ مسئلہ ہو تو مسترد کرنے کے بجائے سپورٹ کو بتائیں۔",
            "قبول اور جلد ڈیلیور ہونے والے آرڈرز آپ کا لیول بڑھاتے ہیں۔",
        ]),
        "bn": ("⚠️ বারবার রিজেক্ট করলে আপনার অ্যাকাউন্ট ঝুঁকিতে পড়ে", [
            "আপনি {rejected}টি টাস্ক রিজেক্ট করেছেন।",
            "বারবার রিজেক্ট করলে অ্যাকাউন্ট স্থগিত হতে পারে এবং টাকা কাটা যেতে পারে।",
            "শুধু যে অর্ডার করতে পারবেন সেটাই গ্রহণ করুন; সমস্যা হলে রিজেক্ট না করে সাপোর্টকে জানান।",
            "গৃহীত ও দ্রুত ডেলিভারি করা অর্ডার আপনার লেভেল বাড়ায়।",
        ]),
    },
    "accept": {
        "ar": ("⚠️ نسبة القبول منخفضة", [
            "نسبة القبول عندك {acceptance}%.",
            "رفض الطلبات بشكل متكرر قد يعرّض حسابك للإيقاف، وقد يتم الخصم عليك مبالغ مالية.",
            "اقبل الطلبات بسرعة، وإذا كان عندك ظرف بلّغ الدعم بدل تجاهل الطلب.",
            "القبول العالي يرفع الـ Segment ويزيد الطلبات اللي تجيك.",
        ]),
        "en": ("⚠️ Your acceptance rate is low", [
            "Your acceptance rate is {acceptance}%.",
            "Repeatedly rejecting orders can get your account suspended and money may be deducted.",
            "Accept orders quickly; if you have a problem, contact support instead of ignoring the order.",
            "A high acceptance rate raises your Segment and brings you more orders.",
        ]),
        "ur": ("⚠️ آپ کی قبولیت کی شرح کم ہے", [
            "آپ کی قبولیت کی شرح {acceptance}% ہے۔",
            "بار بار آرڈر مسترد کرنے سے اکاؤنٹ معطل ہو سکتا ہے اور رقم کٹ سکتی ہے۔",
            "آرڈر جلدی قبول کریں؛ مسئلہ ہو تو آرڈر نظر انداز کرنے کے بجائے سپورٹ کو بتائیں۔",
            "زیادہ قبولیت سے آپ کا سیگمنٹ بڑھتا ہے۔",
        ]),
        "bn": ("⚠️ আপনার অ্যাক্সেপ্টেন্স রেট কম", [
            "আপনার অ্যাক্সেপ্টেন্স রেট {acceptance}%।",
            "বারবার অর্ডার রিজেক্ট করলে অ্যাকাউন্ট স্থগিত হতে পারে এবং টাকা কাটা যেতে পারে।",
            "দ্রুত অর্ডার গ্রহণ করুন; সমস্যা হলে অর্ডার উপেক্ষা না করে সাপোর্টকে জানান।",
            "উচ্চ অ্যাক্সেপ্টেন্স রেট আপনার সেগমেন্ট বাড়ায়।",
        ]),
    },
    "cancel": {
        "ar": ("🚫 تجنّب الإلغاء بسبب مشاكل التوصيل", [
            "عندك {cancel} طلب ملغي ونسبة الإلغاء {cancel_rate}%.",
            "قبل أي إلغاء: اتصل بالعميل أكثر من مرة وراسله في الشات وانتظر المدة المطلوبة.",
            "تأكد من العنوان والموقع على الخريطة قبل التحرك، واسأل العميل لو في أي غموض.",
            "إذا ما رد العميل، صوّر المبنى ولقطة الاتصالات كإثبات، ولا تلغي إلا بعد الرجوع للدعم.",
            "الإلغاء المتكرر يؤثر على مستواك وقد يعرضك لخصومات.",
        ]),
        "en": ("🚫 Avoid cancellations caused by delivery issues", [
            "You have {cancel} cancelled orders and a cancellation rate of {cancel_rate}%.",
            "Before cancelling: call the customer more than once, message in chat, and wait the required time.",
            "Check the address and the map location before moving, and ask the customer if anything is unclear.",
            "If the customer doesn't answer, photograph the building and your call log as proof, and cancel only after contacting support.",
            "Repeated cancellations hurt your level and may lead to deductions.",
        ]),
        "ur": ("🚫 ڈیلیوری مسائل کی وجہ سے منسوخی سے بچیں", [
            "آپ کے {cancel} آرڈرز منسوخ ہیں اور منسوخی کی شرح {cancel_rate}% ہے۔",
            "منسوخ کرنے سے پہلے: کسٹمر کو کئی بار کال کریں، چیٹ میں پیغام بھیجیں اور مطلوبہ وقت انتظار کریں۔",
            "روانہ ہونے سے پہلے ایڈریس اور نقشے کی لوکیشن چیک کریں۔",
            "کسٹمر جواب نہ دے تو عمارت اور کال لاگ کی تصویر ثبوت کے طور پر لیں، اور سپورٹ سے رابطے کے بعد ہی منسوخ کریں۔",
            "بار بار منسوخی آپ کے لیول کو متاثر کرتی ہے اور کٹوتی ہو سکتی ہے۔",
        ]),
        "bn": ("🚫 ডেলিভারি সমস্যার কারণে বাতিল এড়িয়ে চলুন", [
            "আপনার {cancel}টি অর্ডার বাতিল হয়েছে এবং বাতিলের হার {cancel_rate}%।",
            "বাতিল করার আগে: কাস্টমারকে একাধিকবার ফোন করুন, চ্যাটে মেসেজ দিন এবং নির্ধারিত সময় অপেক্ষা করুন।",
            "রওনা দেওয়ার আগে ঠিকানা ও ম্যাপের লোকেশন যাচাই করুন।",
            "কাস্টমার সাড়া না দিলে ভবনের ছবি ও কল লগ প্রমাণ হিসেবে রাখুন, এবং সাপোর্টের সাথে কথা বলেই বাতিল করুন।",
            "বারবার বাতিল করলে আপনার লেভেল কমে এবং টাকা কাটা যেতে পারে।",
        ]),
    },
    "fail": {
        "ar": ("❌ الطلبات الفاشلة", [
            "عندك {failed} طلب فاشل ودرجة الفشل {fail_rate}% (المطلوب 100%).",
            "لا تترك أي طلب بدون تسليم؛ تواصل مع العميل والدعم قبل أي قرار.",
            "سلّم الطلب للعميل وصوّر إثبات التسليم دائمًا.",
        ]),
        "en": ("❌ Failed orders", [
            "You have {failed} failed orders and a fail score of {fail_rate}% (target 100%).",
            "Never leave an order undelivered; contact the customer and support before any decision.",
            "Hand the order to the customer and always photograph the proof of delivery.",
        ]),
        "ur": ("❌ ناکام آرڈرز", [
            "آپ کے {failed} آرڈرز ناکام ہیں اور اسکور {fail_rate}% ہے (ہدف 100%)۔",
            "کوئی آرڈر بغیر ڈیلیوری نہ چھوڑیں؛ فیصلے سے پہلے کسٹمر اور سپورٹ سے رابطہ کریں۔",
            "آرڈر کسٹمر کو دیں اور ڈیلیوری کا ثبوت ہمیشہ تصویر میں رکھیں۔",
        ]),
        "bn": ("❌ ব্যর্থ অর্ডার", [
            "আপনার {failed}টি ব্যর্থ অর্ডার এবং স্কোর {fail_rate}% (লক্ষ্য 100%)।",
            "কোনো অর্ডার ডেলিভারি ছাড়া ফেলে রাখবেন না; সিদ্ধান্তের আগে কাস্টমার ও সাপোর্টের সাথে যোগাযোগ করুন।",
            "অর্ডার কাস্টমারকে দিন এবং ডেলিভারির প্রমাণের ছবি সবসময় রাখুন।",
        ]),
    },
    "verify": {
        "ar": ("🔐 ارفع نسبة التحقق", [
            "نسبة التحقق عندك {verification}% والمطلوب 100%.",
            "أكمل خطوة التحقق/تأكيد التسليم في التطبيق لكل طلب فور التسليم.",
            "تأكد من تسليم الطلب لصاحب الطلب نفسه.",
        ]),
        "en": ("🔐 Raise your verification rate", [
            "Your verification rate is {verification}%; the target is 100%.",
            "Complete the verification / delivery confirmation step in the app for every order right after delivery.",
            "Make sure you hand the order to the person who placed it.",
        ]),
        "ur": ("🔐 اپنی تصدیق کی شرح بڑھائیں", [
            "آپ کی تصدیق کی شرح {verification}% ہے؛ ہدف 100% ہے۔",
            "ہر آرڈر کی ڈیلیوری کے فوراً بعد ایپ میں تصدیق کا مرحلہ مکمل کریں۔",
            "آرڈر اسی شخص کو دیں جس نے آرڈر کیا ہے۔",
        ]),
        "bn": ("🔐 ভেরিফিকেশন রেট বাড়ান", [
            "আপনার ভেরিফিকেশন রেট {verification}%; লক্ষ্য 100%।",
            "প্রতিটি অর্ডার ডেলিভারির পরপরই অ্যাপে ভেরিফিকেশন ধাপ সম্পন্ন করুন।",
            "যিনি অর্ডার করেছেন তাকেই অর্ডার হস্তান্তর করুন।",
        ]),
    },
    "chat": {
        "ar": ("💬 تواصل مع العميل", [
            "نسبة تواصلك مع العميل {chat}%.",
            "افتح الشات في كل طلب وأرسل الرسائل الثلاث: عند المطعم، بعد الاستلام، وعند الوصول.",
        ]),
        "en": ("💬 Talk to the customer", [
            "Your chat-with-customer rate is {chat}%.",
            "Open the chat on every order and send the three messages: at the restaurant, after pickup, and on arrival.",
        ]),
        "ur": ("💬 کسٹمر سے رابطہ کریں", [
            "کسٹمر سے آپ کی چیٹ کی شرح {chat}% ہے۔",
            "ہر آرڈر میں چیٹ کھولیں اور تین پیغام بھیجیں: ریسٹورنٹ پر، آرڈر لینے کے بعد، اور پہنچنے پر۔",
        ]),
        "bn": ("💬 কাস্টমারের সাথে যোগাযোগ করুন", [
            "কাস্টমারের সাথে আপনার চ্যাট রেট {chat}%।",
            "প্রতিটি অর্ডারে চ্যাট খুলুন এবং তিনটি মেসেজ দিন: রেস্টুরেন্টে, অর্ডার নেওয়ার পর, এবং পৌঁছে।",
        ]),
    },
    "low": {
        "ar": ("📉 عدد طلباتك قليل", [
            "طلباتك {orders} والمطلوب {expected} (ناقصك {missing}).",
            "افتح التطبيق أونلاين في أوقات الذروة: الغداء 12–4 والعشاء 7–12.",
            "قف قرب مناطق المطاعم المزدحمة بدل الأماكن البعيدة.",
            "اقبل الطلبات بسرعة وتجنب الرفض عشان تصلك طلبات أكثر.",
            "زد عدد ساعات وأيام عملك في الأسبوع.",
        ]),
        "en": ("📉 Your order count is low", [
            "You have {orders} orders and the required number is {expected} (missing {missing}).",
            "Go online during peak hours: lunch 12–4 PM and dinner 7 PM–12 AM.",
            "Stay near busy restaurant areas instead of remote spots.",
            "Accept orders quickly and avoid rejecting so you receive more orders.",
            "Increase your working hours and working days per week.",
        ]),
        "ur": ("📉 آپ کے آرڈرز کم ہیں", [
            "آپ کے {orders} آرڈرز ہیں اور مطلوبہ تعداد {expected} ہے ({missing} کم)۔",
            "پیک اوقات میں آن لائن رہیں: دوپہر 12–4 اور رات 7–12۔",
            "دور دراز جگہوں کے بجائے مصروف ریسٹورنٹ علاقوں کے قریب رہیں۔",
            "آرڈر جلدی قبول کریں اور مسترد کرنے سے بچیں تاکہ زیادہ آرڈرز ملیں۔",
            "اپنے کام کے گھنٹے اور دن بڑھائیں۔",
        ]),
        "bn": ("📉 আপনার অর্ডার সংখ্যা কম", [
            "আপনার অর্ডার {orders}টি এবং প্রয়োজন {expected}টি ({missing}টি কম)।",
            "ব্যস্ত সময়ে অনলাইন থাকুন: দুপুর ১২–৪ এবং রাত ৭–১২।",
            "দূরের জায়গার বদলে ব্যস্ত রেস্টুরেন্ট এলাকার কাছে থাকুন।",
            "দ্রুত অর্ডার গ্রহণ করুন এবং রিজেক্ট এড়িয়ে চলুন যাতে বেশি অর্ডার পান।",
            "সপ্তাহে কাজের ঘণ্টা ও দিন বাড়ান।",
        ]),
    },
    "proof": {
        "ar": ("📸 دايمًا خلّي معاك إثبات", [
            "صوّر الطلب، ولقطة شاشة للمسافة والشات، وصورة الزحمة والمبنى.",
            "لو جاتلك مخالفة أو الشركة أرسلت لك إيميل، الإثبات هو اللي يحميك.",
            "احتفظ بالصور والمحادثات حتى نهاية الشهر على الأقل.",
        ]),
        "en": ("📸 Always keep proof", [
            "Photograph the order, screenshot the distance and chat, and photograph traffic and the building.",
            "If you get a violation or the company emails you, your proof is what protects you.",
            "Keep the photos and chats until the end of the month at least.",
        ]),
        "ur": ("📸 ہمیشہ ثبوت اپنے پاس رکھیں", [
            "آرڈر کی تصویر، فاصلے اور چیٹ کا اسکرین شاٹ، اور ٹریفک و عمارت کی تصویر لیں۔",
            "اگر خلاف ورزی ملے یا کمپنی ای میل بھیجے تو آپ کا ثبوت ہی آپ کی حفاظت کرتا ہے۔",
            "تصاویر اور چیٹس کم از کم مہینے کے آخر تک محفوظ رکھیں۔",
        ]),
        "bn": ("📸 সবসময় প্রমাণ রাখুন", [
            "অর্ডারের ছবি, দূরত্ব ও চ্যাটের স্ক্রিনশট এবং যানজট ও ভবনের ছবি তুলুন।",
            "জরিমানা বা কোম্পানির ইমেইল এলে আপনার প্রমাণই আপনাকে রক্ষা করবে।",
            "ছবি ও চ্যাট অন্তত মাসের শেষ পর্যন্ত সংরক্ষণ করুন।",
        ]),
    },
    "good": {
        "ar": ("🌟 أداء ممتاز", [
            "أداؤك ممتاز حاليًا، استمر على نفس المستوى.",
            "كمّل بتصوير الإثباتات وإرسال رسائل الشات في كل طلب.",
        ]),
        "en": ("🌟 Excellent performance", [
            "Your performance is excellent right now — keep it up.",
            "Keep taking proof photos and sending chat messages on every order.",
        ]),
        "ur": ("🌟 بہترین کارکردگی", [
            "آپ کی کارکردگی فی الحال بہترین ہے، اسی طرح جاری رکھیں۔",
            "ہر آرڈر پر ثبوت کی تصاویر اور چیٹ پیغامات بھیجتے رہیں۔",
        ]),
        "bn": ("🌟 চমৎকার পারফরম্যান্স", [
            "আপনার পারফরম্যান্স এখন চমৎকার, এভাবেই চালিয়ে যান।",
            "প্রতিটি অর্ডারে প্রমাণের ছবি তুলুন এবং চ্যাটে মেসেজ দিন।",
        ]),
    },
}

SEV_CSS = {"critical": "co-crit", "warn": "co-warn", "info": "co-info", "good": "co-good"}


# ---------------- الرسائل اليدوية (من الأدمن) ----------------
def add_note(project, scope, rider_id, rule, title, body):
    conn = get_conn()
    conn.execute(
        """INSERT INTO coach_notes (project, scope, rider_id, rule, title, body, active, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
        (project, scope, rider_id, rule, title.strip(), body.strip(), datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()


def list_notes(project=None, rider_id=None):
    q, params = "SELECT * FROM coach_notes WHERE 1=1", []
    if project:
        q += " AND project = ?"
        params.append(project)
    if rider_id:
        q += " AND scope = 'rider' AND rider_id = ?"
        params.append(str(rider_id))
    q += " ORDER BY id DESC"
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


def set_note_active(note_id, active):
    conn = get_conn()
    conn.execute("UPDATE coach_notes SET active = ? WHERE id = ?", (1 if active else 0, int(note_id)))
    conn.commit()
    conn.close()


def delete_note(note_id):
    conn = get_conn()
    conn.execute("DELETE FROM coach_notes WHERE id = ?", (int(note_id),))
    conn.commit()
    conn.close()


def notes_for(project, rider_id, triggered):
    out = []
    for n in list_notes(project=project):
        if not n["active"]:
            continue
        if n["scope"] == "all":
            out.append(n)
        elif n["scope"] == "rider" and str(n["rider_id"]) == str(rider_id):
            out.append(n)
        elif n["scope"] == "rule" and n["rule"] in triggered:
            out.append(n)
    return out


def render_coaching(items, notes, lang):
    if not items and not notes:
        return
    CU = COACH_UI[lang]
    e = html.escape
    dr = "rtl" if lang in RTL_LANGS else "ltr"
    parts = []
    for n in notes:
        parts.append(
            '<div class="co-card co-note"><div class="co-head"><span class="co-title">📩 '
            + e(n.get("title") or CU["mgmt"]) + '</span><span class="co-badge">' + e(CU["mgmt"])
            + '</span></div><div class="co-body">' + e(n["body"]).replace("\n", "<br>") + "</div></div>")
    for key, sev, ctx in items:
        title, lines = COACH[key][lang]
        li = "".join("<li>" + e(l.format_map(_Safe(ctx))) + "</li>" for l in lines)
        parts.append(
            '<div class="co-card ' + SEV_CSS[sev] + '"><div class="co-head"><span class="co-title">' + e(title)
            + '</span><span class="co-badge">' + e(CU[sev]) + '</span></div><div class="co-body"><ul>'
            + li + "</ul></div></div>")
    st.markdown('<div class="section-title" dir="' + dr + '">' + e(CU["title"]) + "</div>"
                '<div class="co-wrap" dir="' + dr + '">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def render_rider_coaching(d, project, lang):
    S = load_settings()
    items = evaluate(d, project, S)
    notes = notes_for(project, d["rider_id"], [k for k, _, _ in items])
    render_coaching(items, notes, lang)


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


def hunger_cards(d, seg, T):
    return [
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


def keeta_cards(d, K):
    valid_day = fmt_text(d.get("valid_day"))
    level = fmt_text(d.get("level"))
    return [
        (K["accepted"], fmt_int(d.get("accepted")), None),
        (K["delivered"], fmt_int(d.get("delivered")), None),
        (K["cancel"], fmt_int(d.get("cancel")), None),
        (K["rejected"], fmt_int(d.get("rejected")), None),
        (K["cancel_rate"], fmt_pct(d.get("cancel_rate")), None),
        (K["on_time"], fmt_pct(d.get("on_time_rate")), None),
        (K["late"], fmt_int(d.get("orders_late")), None),
        (K["valid_days"], valid_day, None),
        (K["level"], level, "#14b8a6" if level != "—" else None),
    ]


def render_hunger_body(d, seg, T, lang):
    render_segment_change(d, seg, T, CHANGE[lang])
    render_target_status(d, TARGETMSG[lang])

    st.markdown(f'<div class="section-title">{html.escape(T["overview"])}</div>', unsafe_allow_html=True)
    render_cards(hunger_cards(d, seg, T))
    render_rider_coaching(d, "hunger", lang)
    render_tips(T, TIPS[lang])
    st.markdown(f'<div class="footer-note">{html.escape(T["footer"])}</div>', unsafe_allow_html=True)


def render_keeta_body(d, T, K, lang):
    st.markdown(f'<div class="section-title">{html.escape(T["overview"])}</div>', unsafe_allow_html=True)
    valid_day = fmt_text(d.get("valid_day"))
    level = fmt_text(d.get("level"))
    render_cards(keeta_cards(d, K))
    if valid_day == "—" or level == "—":
        st.markdown(f'<div class="tips-note">ℹ️ {html.escape(K["month_end"])}</div>', unsafe_allow_html=True)
    render_rider_coaching(d, "keeta", lang)
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
        rider_id = normalize_id_text(rider_id)
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
                render_keeta_body(d, T, KTXT[lang], lang)


# ============================================================
# ⚙️ ADMIN CONTROL CENTER
# ============================================================
ADMIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
.block-container {max-width: 1280px !important; padding-top: 1rem !important;}
html, body, [class*="st-"], .adm-hero, .adm-kpi, .adm-sec, .adm-note {font-family: 'Cairo', 'Inter', system-ui, sans-serif;}
.adm-hero {background: radial-gradient(1200px 300px at 90% -20%, rgba(99,102,241,.45), transparent),
           linear-gradient(135deg,#0b1020,#111c3a 60%,#1b2a5c); color:#fff; border-radius:24px; padding:26px 30px; margin:8px 0 18px;
           box-shadow:0 20px 50px rgba(11,16,32,.35); direction:rtl; display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;}
.adm-hero-t {font-size:28px; font-weight:800; letter-spacing:-.3px;}
.adm-hero-s {color:#aab4d4; font-size:14px; margin-top:4px; font-weight:600;}
.adm-chips {display:flex; gap:10px; flex-wrap:wrap;}
.adm-chip {background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.16); border-radius:14px; padding:8px 14px; text-align:center; min-width:96px;}
.adm-chip b {display:block; font-size:20px; font-weight:800;} .adm-chip small {color:#aab4d4; font-size:11px; font-weight:700;}
.adm-kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:6px 0 20px; direction:rtl;}
.adm-kpi {background:#fff; border:1px solid #e8ecf6; border-radius:20px; padding:16px 18px; position:relative; overflow:hidden;
          box-shadow:0 8px 24px rgba(15,23,42,.06); text-align:right;}
.adm-kpi::before {content:""; position:absolute; inset-inline-start:0; top:0; bottom:0; width:5px; background:var(--c);}
.adm-kpi-i {font-size:22px;} .adm-kpi-l {color:#64748b; font-weight:700; font-size:13px; margin-top:2px;}
.adm-kpi-v {font-size:32px; font-weight:800; color:#0f172a; line-height:1.15; margin-top:2px;}
.adm-kpi-s {color:#94a3b8; font-size:12px; font-weight:600;}
.adm-sec {font-size:19px; font-weight:800; color:#0f172a; margin:22px 0 10px; direction:rtl; text-align:right;
          padding-inline-start:12px; border-inline-start:5px solid #6366f1;}
.adm-note {background:#eef2ff; border:1px solid #c7d2fe; color:#3730a3; border-radius:14px; padding:11px 15px; font-weight:600;
           font-size:14px; direction:rtl; text-align:right; margin-bottom:12px;}
.adm-warn {background:#fffbeb; border:1px solid #fde68a; color:#92400e;}
.adm-note-card {background:#fff; border:1px solid #e8ecf6; border-radius:16px; padding:12px 16px; margin-bottom:8px; direction:rtl; text-align:right;
                border-inline-start:5px solid var(--c); box-shadow:0 5px 16px rgba(15,23,42,.05);}
.adm-note-card b {color:#0f172a;} .adm-note-card small {color:#94a3b8; font-weight:600;}
.adm-pill {display:inline-block; background:#f1f5f9; border-radius:999px; padding:2px 10px; font-size:12px; font-weight:700; color:#334155; margin-inline-end:6px;}
.stTabs [data-baseweb="tab-list"] {gap:6px; background:#f1f5f9; padding:6px; border-radius:16px; direction:rtl;}
.stTabs [data-baseweb="tab"] {border-radius:12px; padding:8px 18px; font-weight:700; height:auto;}
.stTabs [aria-selected="true"] {background:#fff !important; box-shadow:0 3px 10px rgba(15,23,42,.10);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
[data-testid="stDataFrame"] {border-radius:16px; overflow:hidden; border:1px solid #e8ecf6;}
</style>
"""

SEV_LABEL = {"critical": "🔴 حرج", "warn": "🟠 تنبيه", "ok": "🟢 سليم"}

NOTE_TEMPLATES = {
    "— رسالة فاضية —": ("", ""),
    "تحذير: الرفض المتكرر": (
        "⚠️ تحذير الرفض المتكرر",
        "الرفض المتكرر للطلبات هيعرّض حسابك للإيقاف وهيتخصم عليك مبالغ مالية. "
        "اقبل بس الطلبات اللي تقدر تنفذها، ولو عندك ظرف بلّغ الدعم قبل ما ترفض."),
    "تنبيه: وقت التوصيل": (
        "⏱️ انتبه لوقت التوصيل",
        "أي تأخير بيأثر على تقييمك. اتحرك فورًا بعد الاستلام، ابعت للعميل رسائل الشات، "
        "ودايمًا خلّي معاك إثبات (صور + لقطات شاشة) عشان لو جاتلك مخالفة أو الشركة بعتتلك إيميل."),
    "حثّ: زيادة الطلبات": (
        "📈 زوّد طلباتك",
        "عدد طلباتك أقل من المطلوب. افتح التطبيق وقت الذروة (الغداء والعشاء)، "
        "وقف قريب من المطاعم المزدحمة، واقبل الطلبات بسرعة."),
    "تذكير: الإثباتات": (
        "📸 دايمًا خلّي معاك إثبات",
        "صوّر الطلب، المسافة، الزحمة، والمبنى، واحتفظ بالشات. الإثبات هو اللي بيحميك لو جاتلك مخالفة."),
}


def adm_kpis(items):
    e = html.escape
    cards = "".join(
        '<div class="adm-kpi" style="--c:' + c + '"><div class="adm-kpi-i">' + ic + '</div><div class="adm-kpi-l">'
        + e(l) + '</div><div class="adm-kpi-v">' + str(v) + '</div><div class="adm-kpi-s">' + e(s) + "</div></div>"
        for ic, l, v, s, c in items)
    st.markdown('<div class="adm-kpis">' + cards + "</div>", unsafe_allow_html=True)


def adm_sec(text):
    st.markdown('<div class="adm-sec">' + html.escape(text) + "</div>", unsafe_allow_html=True)


def adm_note(text, warn=False):
    st.markdown('<div class="adm-note' + (" adm-warn" if warn else "") + '">' + html.escape(text) + "</div>",
                unsafe_allow_html=True)


def build_risk_df(df, project, S):
    """يضيف أعمدة: الحالة / المشاكل / عدد المشاكل / f_<rule> لكل مندوب."""
    sev_list, issues_list, counts = [], [], []
    flags = {k: [] for k in ISSUE_KEYS}
    for rec in df.to_dict("records"):
        items = evaluate(rec, project, S)
        keys = {k for k, _, _ in items}
        sevs = {s for _, s, _ in items}
        sev = "critical" if "critical" in sevs else ("warn" if "warn" in sevs else "ok")
        iss = [k for k in ISSUE_KEYS if k in keys]
        sev_list.append(sev)
        issues_list.append(" + ".join(RULE_AR[k] for k in iss) if iss else "—")
        counts.append(len(iss))
        for k in ISSUE_KEYS:
            flags[k].append(k in keys)
    out = df.copy()
    out["_sev"] = sev_list
    out["الحالة"] = [SEV_LABEL[s] for s in sev_list]
    out["المشاكل"] = issues_list
    out["_n"] = counts
    out["_rank"] = [SEV_RANK.get(s, 3) for s in sev_list]
    for k in ISSUE_KEYS:
        out["f_" + k] = flags[k]
    return out


HUNGER_COLS = {"rider_id": "ID", "rider_name": "الاسم", "segment": "Segment", "orders": "الطلبات",
               "late_orders": "المتأخرة", "on_time_rate": "أون تايم %", "acceptance_rate": "قبول %",
               "verification_rate": "تحقق %", "fail_rate": "Fail %", "الحالة": "الحالة", "المشاكل": "المشاكل"}
KEETA_COLS = {"rider_id": "ID", "rider_name": "الاسم", "delivered": "الموصلة", "rejected": "المرفوضة",
              "cancel": "الملغية", "cancel_rate": "إلغاء %", "on_time_rate": "أون تايم %",
              "orders_late": "المتأخرة", "level": "المستوى", "الحالة": "الحالة", "المشاكل": "المشاكل"}


def table_view(risk, project):
    cmap = HUNGER_COLS if project == "hunger" else KEETA_COLS
    cols = [c for c in cmap if c in risk.columns]
    return risk[cols].rename(columns=cmap)


# ---------------- Dashboard ----------------
def admin_dashboard(project, df, risk, S):
    if df.empty:
        adm_note("لسه مفيش بيانات للمشروع ده. ارفع ملف من تاب 📥 الرفع.", warn=True)
        return
    n = len(risk)
    crit = int((risk["_sev"] == "critical").sum())
    warn = int((risk["_sev"] == "warn").sum())
    ok = n - crit - warn
    ot = risk["on_time_rate"][risk["on_time_rate"] > 0].mean() if "on_time_rate" in risk else 0
    ot = 0 if pd.isna(ot) else ot

    if project == "hunger":
        fs = risk["final_score"][risk["final_score"] > 0].mean()
        adm_kpis([
            ("👥", "المناديب", f"{n:,}", "في آخر ملف", "#6366f1"),
            ("📦", "إجمالي الطلبات", f"{int(risk['orders'].sum()):,}", "كل المناديب", "#0ea5e9"),
            ("⏱️", "متوسط الأون تايم", f"{ot:.1f}%", "المطلوب 98%", "#16a34a" if ot >= 98 else "#f59e0b"),
            ("⭐", "متوسط الجودة", f"{0 if pd.isna(fs) else fs:.1f}%", "Final Score", "#8b5cf6"),
            ("🔴", "حالات حرجة", f"{crit:,}", f"{crit / n * 100:.0f}% من المناديب", "#dc2626"),
            ("🟠", "تحتاج تنبيه", f"{warn:,}", f"{warn / n * 100:.0f}% من المناديب", "#f59e0b"),
        ])
        dist_col, dist_name = "segment", "توزيع الـ Segment"
    else:
        adm_kpis([
            ("👥", "المناديب", f"{n:,}", "في آخر ملف", "#6366f1"),
            ("📦", "المهام الموصلة", f"{int(risk['delivered'].sum()):,}", "كل المناديب", "#0ea5e9"),
            ("⏱️", "متوسط الأون تايم", f"{ot:.1f}%", "Experience On-time", "#16a34a" if ot >= 95 else "#f59e0b"),
            ("⛔", "المهام المرفوضة", f"{int(risk['rejected'].sum()):,}", "إجمالي", "#dc2626"),
            ("🚫", "الملغية", f"{int(risk['cancel'].sum()):,}", "إجمالي", "#f97316"),
            ("🔴", "حالات حرجة", f"{crit:,}", f"{crit / n * 100:.0f}% من المناديب", "#dc2626"),
        ])
        dist_col, dist_name = "level", "توزيع المستويات"

    c1, c2 = st.columns(2)
    with c1:
        adm_sec("أكتر المشاكل انتشارًا")
        counts = {RULE_AR[k]: int(risk["f_" + k].sum()) for k in PROJECT_RULES[project]}
        st.bar_chart(pd.Series(counts, name="عدد المناديب"))
    with c2:
        adm_sec(dist_name)
        dist = risk[dist_col].replace({"": "—"}).value_counts()
        st.bar_chart(dist)

    adm_sec("🚨 أخطر 15 مندوب")
    top = risk.sort_values(["_rank", "_n"], ascending=[True, False]).head(15)
    top = top[top["_sev"] != "ok"]
    if top.empty:
        adm_note("🎉 مفيش مناديب فيهم مشاكل حاليًا.")
    else:
        st.dataframe(table_view(top, project), use_container_width=True, hide_index=True)
    st.caption(f"سليم: {ok:,} | تنبيه: {warn:,} | حرج: {crit:,} — الحدود قابلة للتعديل من تاب 🎯 التعليمات.")


# ---------------- Riders ----------------
def admin_rider_detail(project, d, S):
    name = (d.get("rider_name") or "").strip() or "—"
    if project == "hunger":
        seg = fmt_text(d.get("segment"))
        render_profile(d, name, "Rider ID", "Segment", seg, segment_color(seg))
        render_cards(hunger_cards(d, seg, TEXTS["ar"]))
    else:
        level = fmt_text(d.get("level"))
        render_profile(d, name, "Rider ID", "Level", level, "#14b8a6" if level != "—" else "#64748b")
        render_cards(keeta_cards(d, KTXT["ar"]))

    adm_sec("🧠 التعليمات اللي بتظهر للمندوب ده")
    lang_pick = st.radio("لغة المعاينة", list(LANGS.keys()), index=1, horizontal=True, key="prev_lang")
    items = evaluate(d, project, S)
    render_coaching(items, notes_for(project, d["rider_id"], [k for k, _, _ in items]), LANGS[lang_pick])

    adm_sec("✍️ أضف رسالة يدوية للمندوب ده")
    ver = st.session_state.get("rd_ver", 0)
    tpl = st.selectbox("قالب جاهز", list(NOTE_TEMPLATES.keys()), key=f"rd_tpl_{ver}")
    t0, b0 = NOTE_TEMPLATES[tpl]
    title = st.text_input("العنوان", value=t0, key=f"rd_t_{ver}_{tpl}")
    body = st.text_area("نص التعليمات", value=b0, height=110, key=f"rd_b_{ver}_{tpl}")
    if st.button("📤 إرسال للمندوب", type="primary", key=f"rd_send_{ver}"):
        if not body.strip():
            st.error("اكتب نص التعليمات الأول.")
        else:
            add_note(project, "rider", str(d["rider_id"]), None, title, body)
            st.session_state["rd_ver"] = ver + 1
            st.success("✅ تم. الرسالة هتظهر للمندوب في صفحته.")
            st.rerun()

    mine = list_notes(project=project, rider_id=d["rider_id"])
    if mine:
        adm_sec("رسائل سابقة لنفس المندوب")
        for nt in mine:
            note_row(nt)


def note_row(nt):
    e = html.escape
    scope_txt = {"rider": "مندوب محدد: " + str(nt.get("rider_id")), "all": "كل المناديب",
                 "rule": "كل من عنده: " + RULE_AR.get(nt.get("rule"), str(nt.get("rule")))}[nt["scope"]]
    color = "#9333ea" if nt["active"] else "#cbd5e1"
    st.markdown(
        '<div class="adm-note-card" style="--c:' + color + '"><span class="adm-pill">'
        + ("🟢 فعّالة" if nt["active"] else "⚪ موقوفة") + '</span><span class="adm-pill">' + e(scope_txt)
        + '</span><b>' + e(nt.get("title") or "—") + "</b><div>" + e(nt["body"]).replace("\n", "<br>")
        + "</div><small>#" + str(nt["id"]) + " · " + e(str(nt.get("created_at") or "")) + "</small></div>",
        unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("⏯ إيقاف/تفعيل", key=f"nt_tg_{nt['id']}"):
        set_note_active(nt["id"], not nt["active"])
        st.rerun()
    if b2.button("🗑 حذف", key=f"nt_del_{nt['id']}"):
        delete_note(nt["id"])
        st.rerun()


def admin_riders(project, df, risk, S):
    if df.empty:
        adm_note("لسه مفيش بيانات للمشروع ده.", warn=True)
        return
    f1, f2, f3 = st.columns([2, 2, 3])
    q = f1.text_input("🔎 بحث بالـ ID أو الاسم", key="rd_q").strip().lower()
    sev_pick = f2.selectbox("الحالة", ["الكل", "🔴 حرج", "🟠 تنبيه", "🟢 سليم"], key="rd_sev")
    issue_pick = f3.multiselect("فلتر بالمشكلة", [RULE_AR[k] for k in PROJECT_RULES[project]], key="rd_iss")

    shown = risk
    if q:
        shown = shown[shown["rider_id"].astype(str).str.lower().str.contains(q, regex=False)
                      | shown["rider_name"].astype(str).str.lower().str.contains(q, regex=False)]
    if sev_pick != "الكل":
        shown = shown[shown["الحالة"] == sev_pick]
    for k in PROJECT_RULES[project]:
        if RULE_AR[k] in issue_pick:
            shown = shown[shown["f_" + k]]
    shown = shown.sort_values(["_rank", "_n"], ascending=[True, False])

    st.caption(f"النتائج: {len(shown):,} من {len(risk):,}")
    st.dataframe(table_view(shown, project), use_container_width=True, hide_index=True, height=380)
    st.download_button("📥 تحميل النتائج CSV", table_view(shown, project).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{project}_riders_filtered.csv", mime="text/csv", key="rd_dl")

    adm_sec("🔎 ملف المندوب")
    rid = st.text_input("اكتب ID المندوب لعرض ملفه الكامل", key="rd_pick",
                        value=str(shown["rider_id"].iloc[0]) if len(shown) == 1 else "")
    rid = normalize_id_text(rid)
    if rid:
        d = get_rider(rid, project)
        if d is None:
            st.error("مفيش مندوب بالـ ID ده.")
        else:
            admin_rider_detail(project, d, S)


# ---------------- Coaching ----------------
def admin_coaching(project, df, risk, S):
    left, right = st.columns([3, 2])

    with left:
        adm_sec("✍️ إرسال تعليمات يدوية")
        ver = st.session_state.get("co_ver", 0)
        scope_lbl = st.radio("الإرسال إلى", ["مندوب محدد", "كل المناديب", "كل مندوب عنده مشكلة معينة"],
                             key=f"co_scope_{ver}", horizontal=True)
        rid, rule = None, None
        if scope_lbl == "مندوب محدد":
            rid = normalize_id_text(st.text_input("ID المندوب", key=f"co_rid_{ver}"))
        elif scope_lbl == "كل مندوب عنده مشكلة معينة":
            rule_lbl = st.selectbox("المشكلة", [RULE_AR[k] for k in PROJECT_RULES[project]], key=f"co_rule_{ver}")
            rule = next(k for k, v in RULE_AR.items() if v == rule_lbl)
            if not df.empty:
                st.caption(f"هتوصل لـ {int(risk['f_' + rule].sum()):,} مندوب حاليًا (وأي مندوب يدخل المشكلة دي بعد كده).")
        elif not df.empty:
            st.caption(f"هتوصل لكل المناديب ({len(df):,}).")

        tpl = st.selectbox("قالب جاهز", list(NOTE_TEMPLATES.keys()), key=f"co_tpl_{ver}")
        t0, b0 = NOTE_TEMPLATES[tpl]
        title = st.text_input("العنوان", value=t0, key=f"co_t_{ver}_{tpl}")
        body = st.text_area("نص التعليمات", value=b0, height=130, key=f"co_b_{ver}_{tpl}")
        if st.button("📤 نشر التعليمات", type="primary", use_container_width=True, key=f"co_send_{ver}"):
            if not body.strip():
                st.error("اكتب نص التعليمات الأول.")
            elif scope_lbl == "مندوب محدد" and not rid:
                st.error("اكتب ID المندوب.")
            else:
                scope = {"مندوب محدد": "rider", "كل المناديب": "all"}.get(scope_lbl, "rule")
                add_note(project, scope, rid, rule, title, body)
                st.session_state["co_ver"] = ver + 1
                st.success("✅ تم النشر.")
                st.rerun()

    with right:
        adm_sec("🧠 قواعد التوجيه التلقائي")
        adm_note("التعليمات التلقائية بتتولد لكل مندوب من أرقامه (تأخير، رفض، إلغاء، طلبات قليلة...). عدّل الحدود هنا.")
        new = copy.deepcopy(S)
        for key, label in SETTING_LABELS[project].items():
            cur = S[project][key]
            if isinstance(DEFAULT_SETTINGS[project][key], int):
                new[project][key] = st.number_input(label, min_value=0, value=int(cur), step=1, key=f"st_{project}_{key}")
            else:
                new[project][key] = st.number_input(label, min_value=0.0, value=float(cur), step=0.5, key=f"st_{project}_{key}")
        active = st.multiselect("القواعد المفعّلة", [RULE_AR[k] for k in PROJECT_RULES[project]],
                                default=[RULE_AR[k] for k in PROJECT_RULES[project]
                                         if k not in S["disabled"].get(project, [])],
                                key=f"st_on_{project}")
        new["disabled"][project] = [k for k in PROJECT_RULES[project] if RULE_AR[k] not in active]
        if st.button("💾 حفظ القواعد", use_container_width=True, key=f"st_save_{project}"):
            save_settings(new)
            st.success("✅ تم الحفظ. التعليمات اتحدّثت لكل المناديب فورًا.")
            st.rerun()

    adm_sec("📚 الرسائل اليدوية المنشورة")
    notes = list_notes(project=project)
    if not notes:
        adm_note("مفيش رسائل يدوية لسه.")
    for nt in notes[:30]:
        note_row(nt)


# ---------------- Upload ----------------
def admin_upload_hunger():
    adm_note("⚠️ أي رفع جديد لهنجر ستيشن هيمسح بيانات هنجر القديمة (كيتا مش هتتأثر).", warn=True)
    uploaded = st.file_uploader("Excel / CSV file", type=["xlsx", "xls", "csv"], key="up_hunger")
    if not uploaded:
        return

    try:
        raw, hdr = read_table(uploaded, FIELDS)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    st.success(f"File loaded — {len(raw):,} rows, {len(raw.columns)} columns.")
    if hdr > 0:
        st.info(f"تم اكتشاف صف العناوين تلقائيًا في الصف رقم {hdr + 1}.")
    with st.expander("الأعمدة اللي اتقرأت من الملف"):
        st.write(" | ".join(map(str, raw.columns)))

    mapping = auto_map(raw.columns, FIELDS)
    mapping = manual_map(raw, mapping, FIELDS, "hunger")
    if mapping["rider_id"] is None:
        st.error("مفيش عمود ID. اختاره يدويًا من «مطابقة الأعمدة يدويًا» فوق.")
        return
    missing = [FIELDS[k][0] for k, v in mapping.items() if v is None]
    if missing:
        st.warning("أعمدة مش موجودة (هتظهر 0 / —): " + "، ".join(missing))

    clean = build_clean_df(raw, mapping)
    if clean.empty:
        st.error("مفيش صفوف صالحة بعد القراءة. اتأكد إن عمود الـ ID صح.")
        return

    seg_counts = clean["segment"].value_counts().to_dict()
    st.caption("Segment في الملف: " + " | ".join(f"{k}: {v}" for k, v in seg_counts.items()))
    st.dataframe(clean.drop(columns=["uploaded_at"]).head(20), use_container_width=True, hide_index=True)

    adm_sec("🎯 إعدادات التارجت")
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
    adm_note("⚠️ أي رفع جديد لكيتا هيمسح بيانات كيتا القديمة (هنجر ستيشن مش هتتأثر).", warn=True)
    st.caption("التيمبلت: id | name | accepted | Delivered Tasks | cancel | Rejected Tasks | "
               "Cancellation Rate from Delivery Issues | Experience_On-time Rate (D) | orders late | valid day | level "
               "— الـ valid day والـ level ممكن يكونوا فاضيين لحد آخر الشهر.")

    uploaded = st.file_uploader("Keeta Excel / CSV file", type=["xlsx", "xls", "csv"], key="up_keeta")
    if not uploaded:
        return

    try:
        raw, hdr = read_table(uploaded, KEETA_FIELDS)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    st.success(f"File loaded — {len(raw):,} rows, {len(raw.columns)} columns.")
    if hdr > 0:
        st.info(f"تم اكتشاف صف العناوين تلقائيًا في الصف رقم {hdr + 1}.")
    with st.expander("الأعمدة اللي اتقرأت من الملف"):
        st.write(" | ".join(map(str, raw.columns)))

    mapping = auto_map(raw.columns, KEETA_FIELDS)
    mapping = manual_map(raw, mapping, KEETA_FIELDS, "keeta")
    if mapping["rider_id"] is None:
        st.error("مفيش عمود ID متعرّف. اختاره يدويًا من «مطابقة الأعمدة يدويًا» فوق (الأعمدة اللي اتقرأت ظاهرة في القايمة).")
        return

    missing = [KEETA_FIELDS[k][0] for k, v in mapping.items() if v is None]
    if missing:
        st.warning("أعمدة مش موجودة في الملف (هتظهر 0 / —): " + "، ".join(missing))

    clean = build_keeta_df(raw, mapping)
    if clean.empty:
        st.error("مفيش صفوف صالحة بعد القراءة. اتأكد إن عمود الـ ID صح.")
        return

    no_valid = int((clean["valid_day"] == "").sum())
    no_level = int((clean["level"] == "").sum())
    st.caption(f"الرايدرز: {len(clean):,} | من غير valid day: {no_valid:,} | من غير level: {no_level:,}")
    st.dataframe(clean.drop(columns=["uploaded_at"]).head(20), use_container_width=True, hide_index=True)

    if st.button("💾 مسح القديم ورفع الجديد", type="primary", use_container_width=True, key="save_keeta"):
        replace_keeta_data(clean)
        st.success(f"✅ تم. بيانات كيتا القديمة اتمسحت واتحمّل {len(clean):,} رايدر.")


# ---------------- Admin shell ----------------
def render_admin():
    st.markdown(ADMIN_CSS, unsafe_allow_html=True)
    S = load_settings()
    pend = count_requests("pending")

    pname = st.radio("المشروع", ["🟡 Keeta", "🍔 HungerStation"], horizontal=True, key="adm_proj",
                     label_visibility="collapsed")
    project = "keeta" if "Keeta" in pname else "hunger"

    df = get_all(project)
    risk = build_risk_df(df, project, S) if not df.empty else df
    last = str(df["uploaded_at"].iloc[0]).replace("T", " ") if (not df.empty and "uploaded_at" in df) else "—"
    n_crit = int((risk["_sev"] == "critical").sum()) if not df.empty else 0
    n_notes = len([n for n in list_notes(project=project) if n["active"]])

    st.markdown(
        '<div class="adm-hero"><div><div class="adm-hero-t">Rider Performance · Control Center</div>'
        '<div class="adm-hero-s">' + html.escape(PROJECTS[project]["name"]) + " · آخر رفع: " + html.escape(last)
        + '</div></div><div class="adm-chips">'
        '<div class="adm-chip"><b>' + f"{len(df):,}" + '</b><small>مندوب</small></div>'
        '<div class="adm-chip"><b>' + str(n_crit) + '</b><small>حالة حرجة</small></div>'
        '<div class="adm-chip"><b>' + str(n_notes) + '</b><small>رسالة فعّالة</small></div>'
        '<div class="adm-chip"><b>' + str(pend) + '</b><small>طلب معلّق</small></div>'
        "</div></div>", unsafe_allow_html=True)

    t_dash, t_riders, t_coach, t_up, t_req = st.tabs([
        "📈 لوحة التحكم", "🧑‍✈️ المناديب", "🎯 التعليمات", "📥 الرفع",
        "📨 الطلبات" + (" (" + str(pend) + ")" if pend else "")])

    with t_dash:
        admin_dashboard(project, df, risk, S)
    with t_riders:
        admin_riders(project, df, risk, S)
    with t_coach:
        admin_coaching(project, df, risk, S)
    with t_up:
        if project == "keeta":
            admin_upload_keeta()
        else:
            admin_upload_hunger()
        if not df.empty:
            st.download_button(
                "📥 تحميل البيانات الحالية CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{project}_performance_current.csv", mime="text/csv", key="dl_current")
    with t_req:
        admin_requests_ui()


if page != "Rider Performance":
    if not admin_login():
        st.stop()
    render_admin()
