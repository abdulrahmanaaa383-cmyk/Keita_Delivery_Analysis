"""
🛵 Rider Tracker — نظام تراكنج المناديب (ملف واحد)
=====================================================
كل السيستم في ملف بايثون واحد عشان سهل رفعه على Streamlit Community Cloud.
محتاج جنبه في نفس الـ repo:
  - requirements.txt
  - users.yaml               (بيانات تسجيل الدخول)
  - .streamlit/secrets.toml  (مفاتيح Hunger Station API + Google Sheets — تضيفها لما تجهز)

الأقسام في الملف ده (دور عليها بالتعليقات الكبيرة):
  1) DATABASE (SQLite)          — تخزين المناديب والطلبات
  2) AUTH                       — تسجيل الدخول والصلاحيات
  3) HUNGER STATION API CLIENT  — المكان الوحيد اللي هتفعّله لما توصلك بيانات الـ API
  4) GOOGLE SHEETS SYNC         — تغذية Looker Studio
  5) ANALYSIS                   — كل حسابات الأداء
  6) EXCEL EXPORT                — تصدير كل التقارير في ملف واحد
  7) STREAMLIT UI                — الواجهة نفسها
"""

import sqlite3
import io
from io import BytesIO
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import plotly.express as px

st.set_page_config(page_title="Rider Tracker", page_icon="🛵", layout="wide")

# ════════════════════════════════════════════════════════════════════════
# 1) DATABASE (SQLite) — المصدر الحقيقي للبيانات
# ════════════════════════════════════════════════════════════════════════
DB_PATH = Path(__file__).parent / "data" / "tracker.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS riders (
            rider_id TEXT PRIMARY KEY, name TEXT NOT NULL, phone TEXT, area TEXT,
            vehicle_type TEXT, active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY, rider_id TEXT, rider_name TEXT,
            order_date TEXT NOT NULL, order_time TEXT, pickup_area TEXT, drop_area TEXT,
            distance_km REAL DEFAULT 0, order_value REAL DEFAULT 0, currency TEXT DEFAULT 'SAR',
            status TEXT DEFAULT 'delivered', source TEXT DEFAULT 'manual',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (rider_id) REFERENCES riders(rider_id)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sync_type TEXT, rows_synced INTEGER,
            status TEXT, message TEXT, synced_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def db_upsert_rider(rider: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO riders (rider_id, name, phone, area, vehicle_type, active)
        VALUES (:rider_id, :name, :phone, :area, :vehicle_type, :active)
        ON CONFLICT(rider_id) DO UPDATE SET
            name=excluded.name, phone=excluded.phone, area=excluded.area,
            vehicle_type=excluded.vehicle_type, active=excluded.active
    """, rider)
    conn.commit()
    conn.close()


def db_get_riders() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM riders ORDER BY name", conn)
    conn.close()
    return df


def db_delete_rider(rider_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM riders WHERE rider_id=?", (rider_id,))
    conn.commit()
    conn.close()


def db_upsert_orders(rows: list) -> int:
    if not rows:
        return 0
    conn = get_conn()
    cur = conn.cursor()
    for r in rows:
        r.setdefault("currency", "SAR")
        r.setdefault("status", "delivered")
        r.setdefault("source", "manual")
        cur.execute("""
            INSERT INTO orders (order_id, rider_id, rider_name, order_date, order_time,
                                 pickup_area, drop_area, distance_km, order_value,
                                 currency, status, source)
            VALUES (:order_id, :rider_id, :rider_name, :order_date, :order_time,
                    :pickup_area, :drop_area, :distance_km, :order_value,
                    :currency, :status, :source)
            ON CONFLICT(order_id) DO UPDATE SET
                rider_id=excluded.rider_id, rider_name=excluded.rider_name,
                order_date=excluded.order_date, order_time=excluded.order_time,
                pickup_area=excluded.pickup_area, drop_area=excluded.drop_area,
                distance_km=excluded.distance_km, order_value=excluded.order_value,
                currency=excluded.currency, status=excluded.status, source=excluded.source
        """, r)
    conn.commit()
    conn.close()
    return len(rows)


def db_get_orders(date_from=None, date_to=None) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT * FROM orders WHERE 1=1"
    params = []
    if date_from:
        q += " AND order_date >= ?"; params.append(str(date_from))
    if date_to:
        q += " AND order_date <= ?"; params.append(str(date_to))
    q += " ORDER BY order_date DESC, order_time DESC"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df


def db_delete_order(order_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM orders WHERE order_id=?", (order_id,))
    conn.commit()
    conn.close()


def db_log_sync(sync_type: str, rows: int, status: str, message: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO sync_log (sync_type, rows_synced, status, message) VALUES (?,?,?,?)",
        (sync_type, rows, status, message)
    )
    conn.commit()
    conn.close()


def db_get_sync_log(limit=20) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM sync_log ORDER BY synced_at DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df


# ════════════════════════════════════════════════════════════════════════
# 2) AUTH — تسجيل الدخول والصلاحيات (بدون مكتبات خارجية إضافية)
# ════════════════════════════════════════════════════════════════════════
import yaml
import bcrypt

USERS_FILE = Path(__file__).parent / "users.yaml"


def load_users_config():
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def login_screen():
    """شاشة لوجن بسيطة، بترجع (name, username, role). بتوقف الصفحة لحد ما تسجل دخول صح."""
    if st.session_state.get("auth_ok"):
        return st.session_state["auth_name"], st.session_state["auth_username"], st.session_state["auth_role"]

    st.markdown("""
    <div style="max-width:420px;margin:80px auto 0;text-align:center;">
        <div style="font-size:40px;">🛵</div>
        <div style="font-size:22px;font-weight:800;">Rider Tracker</div>
        <div style="color:#6b7280;font-size:13px;margin-bottom:24px;">تسجيل الدخول للمتابعة</div>
    </div>
    """, unsafe_allow_html=True)

    config = load_users_config()
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        with st.form("login_form"):
            username = st.text_input("اسم المستخدم")
            password = st.text_input("كلمة المرور", type="password")
            submitted = st.form_submit_button("دخول", use_container_width=True)

        if submitted:
            users = config["credentials"]["usernames"]
            user = users.get(username)
            if user and check_password(password, user["password"]):
                st.session_state["auth_ok"] = True
                st.session_state["auth_name"] = user["name"]
                st.session_state["auth_username"] = username
                st.session_state["auth_role"] = user.get("role", "member")
                st.rerun()
            else:
                st.error("❌ اسم المستخدم أو كلمة المرور غلط")

    st.stop()


def require_admin():
    if st.session_state.get("auth_role") != "admin":
        st.warning("🔒 الصفحة دي للأدمن بس.")
        st.stop()


def logout_button():
    if st.sidebar.button("🚪 تسجيل الخروج"):
        for k in ["auth_ok", "auth_name", "auth_username", "auth_role"]:
            st.session_state.pop(k, None)
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# 3) HUNGER STATION API CLIENT — عدّل هنا بس لما توصلك بيانات الـ API
# ════════════════════════════════════════════════════════════════════════
class HungerStationClient:
    def __init__(self):
        cfg = st.secrets.get("hunger_station", {}) if hasattr(st, "secrets") else {}
        self.base_url = cfg.get("base_url", "")
        self.api_key = cfg.get("api_key", "")
        self.is_configured = bool(self.base_url and self.api_key)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def fetch_orders(self, date_from: date, date_to: date) -> list:
        """
        لازم ترجع list of dict بنفس شكل جدول orders:
        order_id, rider_id, rider_name, order_date, order_time,
        pickup_area, drop_area, distance_km, order_value, currency, status

        لما توصلك الـ docs الحقيقية من هنجر ستيشن، فك التعليق عن الكود تحت
        وعدّل أسماء الحقول عشان تطابق شكل الـ JSON بتاعهم بالظبط.
        """
        if not self.is_configured:
            raise RuntimeError(
                "لسه معملتش ربط مع Hunger Station API. ضيف base_url و api_key "
                "في .streamlit/secrets.toml تحت [hunger_station]"
            )
        # import requests
        # resp = requests.get(f"{self.base_url}/v1/orders", headers=self._headers(),
        #     params={"date_from": date_from.isoformat(), "date_to": date_to.isoformat()}, timeout=30)
        # resp.raise_for_status()
        # raw = resp.json().get("data", [])
        # mapped = []
        # for o in raw:
        #     mapped.append({
        #         "order_id": str(o["id"]), "rider_id": str(o["courier"]["id"]),
        #         "rider_name": o["courier"]["name"], "order_date": o["created_at"][:10],
        #         "order_time": o["created_at"][11:16], "pickup_area": o.get("branch_name", ""),
        #         "drop_area": o.get("customer_area", ""), "distance_km": float(o.get("distance_km", 0)),
        #         "order_value": float(o.get("total", 0)), "currency": o.get("currency", "SAR"),
        #         "status": o.get("status", "delivered"),
        #     })
        # return mapped
        raise NotImplementedError("الشكل الحقيقي للـ API لسه مش معروف — عدّل الدالة دي لما توصلك الـ docs.")

    def fetch_riders(self) -> list:
        if not self.is_configured:
            raise RuntimeError("لسه معملتش ربط مع Hunger Station API.")
        raise NotImplementedError("هتتفعّل لما توصل بيانات الـ API.")


# ════════════════════════════════════════════════════════════════════════
# 4) GOOGLE SHEETS SYNC — تغذية Looker Studio
# ════════════════════════════════════════════════════════════════════════
def sheets_is_configured() -> bool:
    try:
        return "gcp_service_account" in st.secrets and bool(st.secrets["gcp_service_account"].get("spreadsheet_id"))
    except Exception:
        return False


def sheets_get_url() -> str:
    try:
        sid = st.secrets["gcp_service_account"]["spreadsheet_id"]
        return f"https://docs.google.com/spreadsheets/d/{sid}"
    except Exception:
        return ""


def _sheets_write_df(sh, tab_name: str, df: pd.DataFrame):
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=max(len(df) + 10, 100), cols=max(len(df.columns) + 2, 10))
    if df.empty:
        ws.update([["لا توجد بيانات"]])
        return
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def sheets_sync_all(orders_df: pd.DataFrame, riders_summary_df: pd.DataFrame, daily_summary_df: pd.DataFrame) -> int:
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_dict = dict(st.secrets["gcp_service_account"])
    spreadsheet_id = creds_dict.pop("spreadsheet_id", None)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sh = client.open_by_key(spreadsheet_id)

    _sheets_write_df(sh, "Orders", orders_df)
    _sheets_write_df(sh, "Riders_Summary", riders_summary_df)
    _sheets_write_df(sh, "Daily_Summary", daily_summary_df)
    return len(orders_df)


# ════════════════════════════════════════════════════════════════════════
# 5) ANALYSIS — كل حسابات الأداء والتحليل
# ════════════════════════════════════════════════════════════════════════
def riders_summary(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=["rider_id", "rider_name", "total_orders", "delivered",
                                      "cancelled", "total_km", "avg_km_per_order",
                                      "total_value", "acceptance_rate"])
    g = orders.groupby(["rider_id", "rider_name"], dropna=False)
    out = g.agg(
        total_orders=("order_id", "count"),
        delivered=("status", lambda s: (s == "delivered").sum()),
        cancelled=("status", lambda s: (s == "cancelled").sum()),
        rejected=("status", lambda s: (s == "rejected").sum()),
        total_km=("distance_km", "sum"),
        total_value=("order_value", "sum"),
    ).reset_index()
    out["avg_km_per_order"] = (out["total_km"] / out["total_orders"]).round(2)
    out["acceptance_rate"] = ((out["delivered"] / out["total_orders"]) * 100).round(1)
    return out.sort_values("total_orders", ascending=False)


def daily_summary(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=["order_date", "total_orders", "total_km", "total_value", "active_riders"])
    g = orders.groupby("order_date")
    out = g.agg(
        total_orders=("order_id", "count"), total_km=("distance_km", "sum"),
        total_value=("order_value", "sum"), active_riders=("rider_id", "nunique"),
    ).reset_index()
    return out.sort_values("order_date", ascending=False)


def hourly_distribution(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty or "order_time" not in orders.columns:
        return pd.DataFrame(columns=["hour", "total_orders"])
    tmp = orders.copy()
    tmp["hour"] = tmp["order_time"].astype(str).str.slice(0, 2)
    out = tmp.groupby("hour").agg(total_orders=("order_id", "count")).reset_index()
    return out.sort_values("hour")


def compute_kpis(orders: pd.DataFrame) -> dict:
    if orders.empty:
        return dict(total_orders=0, total_km=0, total_value=0, active_riders=0, avg_km=0, cancellation_rate=0)
    total = len(orders)
    cancelled = (orders["status"] == "cancelled").sum()
    return dict(
        total_orders=total, total_km=round(orders["distance_km"].sum(), 1),
        total_value=round(orders["order_value"].sum(), 1), active_riders=orders["rider_id"].nunique(),
        avg_km=round(orders["distance_km"].mean(), 2) if total else 0,
        cancellation_rate=round((cancelled / total) * 100, 1) if total else 0,
    )


def auto_insights(orders: pd.DataFrame, r_summary: pd.DataFrame, d_summary: pd.DataFrame) -> list:
    insights = []
    if orders.empty:
        return ["لا توجد بيانات كافية لعمل تحليل بعد."]
    k = compute_kpis(orders)
    insights.append(f"إجمالي الطلبات في الفترة المحددة: {k['total_orders']} طلب، بإجمالي {k['total_km']} كم.")
    insights.append(f"نسبة الإلغاء: {k['cancellation_rate']}% — "
                     f"{'مرتفعة، يُنصح بمراجعة أسباب الإلغاء' if k['cancellation_rate'] > 15 else 'ضمن المعدل الطبيعي'}.")
    if not r_summary.empty:
        top = r_summary.iloc[0]
        insights.append(f"أعلى مندوب في عدد الطلبات: {top['rider_name']} بـ {int(top['total_orders'])} طلب.")
        low_acc = r_summary[r_summary["acceptance_rate"] < 70]
        if not low_acc.empty:
            names = "، ".join(low_acc["rider_name"].head(5).tolist())
            insights.append(f"مناديب بنسبة قبول أقل من 70%: {names}.")
    if not d_summary.empty:
        busiest = d_summary.sort_values("total_orders", ascending=False).iloc[0]
        insights.append(f"أكثر يوم ازدحاماً: {busiest['order_date']} بـ {int(busiest['total_orders'])} طلب.")
    return insights


# ════════════════════════════════════════════════════════════════════════
# 6) EXCEL EXPORT — تصدير كل التقارير في ملف واحد منسّق
# ════════════════════════════════════════════════════════════════════════
HEADER_FILL = PatternFill("solid", start_color="4F46E5")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=11)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
THIN = Side(style="thin", color="D1D5DB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
ZEBRA = PatternFill("solid", start_color="F9FAFB")


def _xl_write_table(ws, df: pd.DataFrame, start_row=1, title=None):
    r = start_row
    if title:
        ws.cell(row=r, column=1, value=title).font = Font(bold=True, size=13, color="1F2937")
        r += 2
    if df.empty:
        ws.cell(row=r, column=1, value="لا توجد بيانات")
        return r + 1
    for ci, col in enumerate(df.columns, 1):
        cell = ws.cell(row=r, column=ci, value=str(col))
        cell.font = HEADER_FONT; cell.fill = HEADER_FILL; cell.alignment = CENTER; cell.border = BORDER
        ws.column_dimensions[get_column_letter(ci)].width = max(14, len(str(col)) + 4)
    ws.row_dimensions[r].height = 22
    for ri, (_, row) in enumerate(df.iterrows(), r + 1):
        for ci, val in enumerate(row, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.border = BORDER; cell.alignment = CENTER
            if ri % 2 == 0:
                cell.fill = ZEBRA
    return r + len(df) + 2


def build_full_report(orders: pd.DataFrame, date_from=None, date_to=None) -> bytes:
    r_sum = riders_summary(orders)
    d_sum = daily_summary(orders)
    k = compute_kpis(orders)
    insights = auto_insights(orders, r_sum, d_sum)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ملخص عام"; ws.sheet_view.rightToLeft = True
    period = f"{date_from} → {date_to}" if date_from and date_to else "كل البيانات"
    ws.cell(row=1, column=1, value=f"تقرير أداء المناديب — الفترة: {period}").font = Font(bold=True, size=15)
    ws.cell(row=2, column=1, value=f"تاريخ إنشاء التقرير: {datetime.now().strftime('%Y-%m-%d %H:%M')}").font = Font(italic=True, size=10, color="6B7280")
    kpi_df = pd.DataFrame([
        ("إجمالي الطلبات", k["total_orders"]), ("إجمالي الكيلومترات", k["total_km"]),
        ("إجمالي القيمة", k["total_value"]), ("عدد المناديب النشطين", k["active_riders"]),
        ("متوسط الكم لكل طلب", k["avg_km"]), ("نسبة الإلغاء %", k["cancellation_rate"]),
    ], columns=["المؤشر", "القيمة"])
    _xl_write_table(ws, kpi_df, start_row=4, title="المؤشرات الرئيسية")

    ws2 = wb.create_sheet("أداء المناديب"); ws2.sheet_view.rightToLeft = True
    r_sum_ar = r_sum.rename(columns={
        "rider_id": "رقم المندوب", "rider_name": "الاسم", "total_orders": "إجمالي الطلبات",
        "delivered": "تم التسليم", "cancelled": "ملغي", "rejected": "مرفوض",
        "total_km": "إجمالي الكم", "total_value": "إجمالي القيمة",
        "avg_km_per_order": "متوسط كم/طلب", "acceptance_rate": "نسبة القبول %"
    })
    _xl_write_table(ws2, r_sum_ar, title="أداء كل مندوب")

    ws3 = wb.create_sheet("ملخص يومي"); ws3.sheet_view.rightToLeft = True
    d_sum_ar = d_sum.rename(columns={
        "order_date": "التاريخ", "total_orders": "إجمالي الطلبات",
        "total_km": "إجمالي الكم", "total_value": "إجمالي القيمة", "active_riders": "عدد المناديب"
    })
    _xl_write_table(ws3, d_sum_ar, title="ملخص كل يوم")

    ws4 = wb.create_sheet("تفاصيل الطلبات"); ws4.sheet_view.rightToLeft = True
    orders_ar = orders.rename(columns={
        "order_id": "رقم الطلب", "rider_id": "رقم المندوب", "rider_name": "اسم المندوب",
        "order_date": "التاريخ", "order_time": "الوقت", "pickup_area": "منطقة الاستلام",
        "drop_area": "منطقة التسليم", "distance_km": "الكيلومترات", "order_value": "القيمة",
        "currency": "العملة", "status": "الحالة", "source": "المصدر"
    }) if not orders.empty else orders
    _xl_write_table(ws4, orders_ar, title="كل الطلبات (بعد الفلاتر)")

    ws5 = wb.create_sheet("تحليل تلقائي"); ws5.sheet_view.rightToLeft = True
    ws5.cell(row=1, column=1, value="ملاحظات وتحليل تلقائي").font = Font(bold=True, size=14)
    for i, line in enumerate(insights, 3):
        c = ws5.cell(row=i, column=1, value=f"•  {line}")
        c.alignment = Alignment(horizontal="right", wrap_text=True)
        ws5.row_dimensions[i].height = 22
    ws5.column_dimensions["A"].width = 100

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════
# 7) STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════
name, username, role = login_screen()
logout_button()
init_db()

st.sidebar.markdown(f"👋 أهلاً **{name}** ({'أدمن' if role == 'admin' else 'عضو فريق'})")

st.sidebar.markdown("### 🔎 الفلاتر")
default_from = date.today() - timedelta(days=30)
date_from = st.sidebar.date_input("من تاريخ", value=default_from)
date_to = st.sidebar.date_input("إلى تاريخ", value=date.today())

all_orders = db_get_orders(date_from=date_from, date_to=date_to)
all_riders_df = db_get_riders()

rider_filter = st.sidebar.multiselect(
    "فلترة بالمندوب",
    options=sorted(all_orders["rider_name"].dropna().unique().tolist()) if not all_orders.empty else []
)
if rider_filter:
    all_orders = all_orders[all_orders["rider_name"].isin(rider_filter)]

status_filter = st.sidebar.multiselect("فلترة بالحالة", options=["delivered", "cancelled", "rejected"], default=[])
if status_filter:
    all_orders = all_orders[all_orders["status"].isin(status_filter)]

st.title("🛵 نظام تراكنج المناديب")

tabs = st.tabs(["📊 الداشبورد", "➕ إدخال / استيراد", "🏍️ المناديب", "📤 التصدير والمزامنة", "⚙️ الإعدادات"])

# ── TAB 1: Dashboard ──────────────────────────────────────────────────────
with tabs[0]:
    if all_orders.empty:
        st.info("مفيش بيانات في الفترة/الفلاتر دي لسه. روح تبويب 'إدخال / استيراد' وضيف بيانات.")
    else:
        k = compute_kpis(all_orders)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("إجمالي الطلبات", k["total_orders"])
        c2.metric("إجمالي الكيلومترات", k["total_km"])
        c3.metric("إجمالي القيمة", f"{k['total_value']:,.0f}")
        c4.metric("عدد المناديب النشطين", k["active_riders"])
        c5.metric("نسبة الإلغاء", f"{k['cancellation_rate']}%")

        r_sum = riders_summary(all_orders)
        d_sum = daily_summary(all_orders)
        h_dist = hourly_distribution(all_orders)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("📈 الطلبات يومياً")
            st.plotly_chart(px.bar(d_sum.sort_values("order_date"), x="order_date", y="total_orders"), use_container_width=True)
        with cc2:
            st.subheader("🕐 توزيع الطلبات على الساعات")
            st.plotly_chart(px.bar(h_dist, x="hour", y="total_orders"), use_container_width=True)

        st.subheader("🏆 أداء المناديب")
        st.dataframe(r_sum, use_container_width=True, height=350)

        st.subheader("🧠 تحليل تلقائي")
        for line in auto_insights(all_orders, r_sum, d_sum):
            st.markdown(f"- {line}")

# ── TAB 2: Data entry / import ───────────────────────────────────────────
with tabs[1]:
    st.markdown("### طريقة إضافة البيانات دلوقتي")
    st.caption("لغاية ما يوصلك API هنجر ستيشن، تقدر تضيف الطلبات يدوي أو تستورد ملف Excel/CSV دفعة واحدة.")

    sub1, sub2, sub3 = st.tabs(["✍️ إدخال يدوي", "📁 استيراد ملف", "🔌 مزامنة من Hunger Station API"])

    with sub1:
        with st.form("manual_order_form", clear_on_submit=True):
            oc1, oc2, oc3 = st.columns(3)
            order_id = oc1.text_input("رقم الطلب *")
            r_name = oc2.text_input("اسم المندوب *")
            r_id = oc3.text_input("رقم المندوب *")
            oc4, oc5, oc6 = st.columns(3)
            o_date = oc4.date_input("التاريخ", value=date.today())
            o_time = oc5.time_input("الوقت")
            km = oc6.number_input("الكيلومترات", min_value=0.0, step=0.1)
            oc7, oc8, oc9 = st.columns(3)
            value = oc7.number_input("قيمة الطلب", min_value=0.0, step=1.0)
            status = oc8.selectbox("الحالة", ["delivered", "cancelled", "rejected"])
            pickup = oc9.text_input("منطقة الاستلام")
            if st.form_submit_button("✅ إضافة الطلب"):
                if not order_id or not r_name or not r_id:
                    st.error("رقم الطلب واسم ورقم المندوب حقول إجبارية.")
                else:
                    db_upsert_orders([{
                        "order_id": order_id, "rider_id": r_id, "rider_name": r_name,
                        "order_date": o_date.isoformat(), "order_time": o_time.strftime("%H:%M"),
                        "pickup_area": pickup, "drop_area": "", "distance_km": km,
                        "order_value": value, "status": status, "source": "manual",
                    }])
                    st.success(f"✅ تم إضافة الطلب {order_id}")
                    st.rerun()

    with sub2:
        st.caption("الملف لازم يحتوي أعمدة: order_id, rider_id, rider_name, order_date, order_time, "
                    "pickup_area, drop_area, distance_km, order_value, status")
        up = st.file_uploader("ارفع ملف Excel أو CSV", type=["xlsx", "csv"])
        if up:
            try:
                df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                st.dataframe(df.head(20), use_container_width=True)
                if st.button("📥 استيراد كل الصفوف دي"):
                    required = ["order_id", "rider_id", "rider_name", "order_date"]
                    missing = [c for c in required if c not in df.columns]
                    if missing:
                        st.error(f"ناقص الأعمدة دي: {missing}")
                    else:
                        rows = df.fillna("").astype(str).to_dict("records")
                        for r in rows:
                            r["distance_km"] = float(r.get("distance_km") or 0)
                            r["order_value"] = float(r.get("order_value") or 0)
                        n = db_upsert_orders(rows)
                        st.success(f"✅ تم استيراد {n} صف بنجاح")
                        st.rerun()
            except Exception as e:
                st.error(f"مشكلة في قراءة الملف: {e}")

    with sub3:
        client = HungerStationClient()
        if not client.is_configured:
            st.warning(
                "🔌 لسه معملتش ربط الـ API. لما توصلك بيانات الدخول من هنجر ستيشن، ضيفهم في "
                "`.streamlit/secrets.toml` تحت `[hunger_station]` — ودالة `fetch_orders` فوق في "
                "الملف ده هتشتغل تلقائياً من غير ما تغيّر حاجة تانية."
            )
        else:
            c1, c2 = st.columns(2)
            f_from = c1.date_input("من", value=date.today(), key="api_from")
            f_to = c2.date_input("إلى", value=date.today(), key="api_to")
            if st.button("🔄 اسحب الطلبات من هنجر ستيشن الآن"):
                try:
                    rows = client.fetch_orders(f_from, f_to)
                    n = db_upsert_orders(rows)
                    db_log_sync("hunger_station_api", n, "success")
                    st.success(f"✅ اتسحب {n} طلب من الـ API")
                    st.rerun()
                except Exception as e:
                    db_log_sync("hunger_station_api", 0, "error", str(e))
                    st.error(f"❌ فشلت المزامنة: {e}")

# ── TAB 3: Riders management ──────────────────────────────────────────────
with tabs[2]:
    st.subheader("🏍️ إدارة المناديب")
    st.dataframe(all_riders_df, use_container_width=True)

    if role == "admin":
        with st.expander("➕ إضافة / تعديل مندوب"):
            with st.form("rider_form", clear_on_submit=True):
                rc1, rc2, rc3 = st.columns(3)
                rid = rc1.text_input("رقم المندوب *")
                rname = rc2.text_input("الاسم *")
                rphone = rc3.text_input("الموبايل")
                rc4, rc5, rc6 = st.columns(3)
                rarea = rc4.text_input("المنطقة")
                rvehicle = rc5.selectbox("نوع المركبة", ["دراجة نارية", "دراجة هوائية", "سيارة"])
                ractive = rc6.selectbox("الحالة", ["نشط", "غير نشط"])
                if st.form_submit_button("💾 حفظ"):
                    if not rid or not rname:
                        st.error("رقم المندوب والاسم إجباريين")
                    else:
                        db_upsert_rider({
                            "rider_id": rid, "name": rname, "phone": rphone, "area": rarea,
                            "vehicle_type": rvehicle, "active": 1 if ractive == "نشط" else 0,
                        })
                        st.success("✅ تم الحفظ")
                        st.rerun()

        with st.expander("🗑️ حذف مندوب"):
            if not all_riders_df.empty:
                to_del = st.selectbox("اختر مندوب للحذف", all_riders_df["rider_id"] + " - " + all_riders_df["name"])
                if st.button("حذف نهائي", type="secondary"):
                    db_delete_rider(to_del.split(" - ")[0])
                    st.success("تم الحذف")
                    st.rerun()
    else:
        st.caption("إضافة/حذف المناديب متاح للأدمن فقط.")

# ── TAB 4: Export & sync ──────────────────────────────────────────────────
with tabs[3]:
    st.subheader("📤 تصدير كل التقارير مرة واحدة")
    st.caption("بيطلع ملف Excel واحد فيه: ملخص عام، أداء المناديب، ملخص يومي، تفاصيل الطلبات، وتحليل تلقائي.")

    if st.button("📊 توليد ملف التقرير الكامل", use_container_width=True):
        st.session_state["last_report"] = build_full_report(all_orders, date_from, date_to)

    if "last_report" in st.session_state:
        st.download_button(
            "📥 تحميل التقرير الكامل (Excel)", data=st.session_state["last_report"],
            file_name=f"rider_tracker_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()
    st.subheader("🔄 مزامنة مع Google Sheets (لتغذية Looker Studio)")
    if not sheets_is_configured():
        st.warning("لسه معملتش ربط Google Sheets. اتبع خطوات README (قسم Google Sheets Setup).")
    else:
        st.info(f"الشيت متصل: {sheets_get_url()}")
        if role == "admin":
            if st.button("🔄 مزامنة البيانات دلوقتي مع Google Sheets"):
                try:
                    n = sheets_sync_all(all_orders, riders_summary(all_orders), daily_summary(all_orders))
                    db_log_sync("google_sheets", n, "success")
                    st.success(f"✅ تمت مزامنة {n} صف مع Google Sheets.")
                except Exception as e:
                    db_log_sync("google_sheets", 0, "error", str(e))
                    st.error(f"❌ فشلت المزامنة: {e}")
        else:
            st.caption("زرار المزامنة متاح للأدمن فقط.")

    st.divider()
    st.subheader("📜 سجل آخر عمليات المزامنة")
    st.dataframe(db_get_sync_log(), use_container_width=True)

# ── TAB 5: Settings ────────────────────────────────────────────────────────
with tabs[4]:
    require_admin()
    st.subheader("⚙️ الإعدادات")
    st.markdown("""
    - **إدارة المستخدمين والصلاحيات:** عدّل `users.yaml` (باسورد مشفّر bcrypt لكل مستخدم).
    - **ربط Hunger Station API:** عدّل `.streamlit/secrets.toml` تحت `[hunger_station]`.
    - **ربط Google Sheets:** عدّل `.streamlit/secrets.toml` تحت `[gcp_service_account]`.
    """)
    if not all_orders.empty:
        del_id = st.selectbox("اختر رقم طلب للحذف", all_orders["order_id"])
        if st.button("🗑️ حذف الطلب المحدد"):
            db_delete_order(del_id)
            st.success("تم الحذف")
            st.rerun()
