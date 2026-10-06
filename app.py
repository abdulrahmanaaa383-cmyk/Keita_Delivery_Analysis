import hmac
import html
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# RIDER PERFORMANCE PORTAL (simplified + flexible)
# - كل رفع جديد بيمسح البيانات القديمة بالكامل
# - بيحتاج 11 عمود بس، وبيتعرف عليهم تلقائيًا أو تختارهم يدوي
# ============================================================

st.set_page_config(page_title="Rider Performance", page_icon="🏆", layout="centered")

DB_PATH = Path(__file__).parent / "rider_performance.db"
TABLE = "rider_stats"

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
    """يمسح القديم ويحط الجديد."""
    conn = get_conn()
    conn.execute("DROP TABLE IF EXISTS performance")  # جداول النسخة القديمة
    conn.execute("DROP TABLE IF EXISTS riders")
    df.to_sql(TABLE, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


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

if page == "Rider Performance":
    lang_name = st.radio("Language", list(LANGS.keys()), horizontal=True,
                         key="lang", label_visibility="collapsed")
    lang = LANGS[lang_name]
    T = TEXTS[lang]

    if lang in RTL_LANGS:
        st.markdown("""
        <style>
            .brand, .profile, .metric-card, .section-title, .footer-note, .tips,
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

    st.title("⚙️ Performance Admin")
    tab_upload, tab_data = st.tabs(["📥 Upload", "📊 Current Data"])

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

                if st.button("💾 مسح القديم ورفع الجديد", type="primary", use_container_width=True):
                    replace_all_data(clean)
                    st.success(f"✅ تم. البيانات القديمة اتمسحت واتحمّل {len(clean):,} رايدر.")

    with tab_data:
        current = get_all()
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
                file_name="rider_performance_current.csv",
                mime="text/csv",
                use_container_width=True,
            )
