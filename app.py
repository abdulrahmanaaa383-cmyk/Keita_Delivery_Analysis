import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timezone, timedelta
from io import BytesIO

# ==================================================================================
# التوقيت السعودي UTC+3
# ==================================================================================
SAUDI_TZ = timezone(timedelta(hours=3))

def now_saudi():
    return datetime.now(SAUDI_TZ)

# ==================================================================================
# إعداد الصفحة
# ==================================================================================
st.set_page_config(
    page_title="تتبع طلبات المناديب",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================================================
# ملفات البيانات
# ==================================================================================
DATA_FILE      = "delivery_data.json"
DRIVERS_FILE   = "drivers.json"
CONFIG_FILE    = "config.json"
REPORTS_FILE   = "daily_reports.json"
SHIFTS_FILE    = "shifts.json"
LOCATIONS_FILE = "driver_locations.json"   # ← مواقع المناديب اللايف

ADMIN_PASSWORD = "admin123"

# ==================================================================================
# دوال مساعدة
# ==================================================================================
def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_today():
    return now_saudi().strftime("%Y-%m-%d")

def get_driver_token(driver_name: str, day: str) -> str:
    import hashlib
    raw = f"{driver_name}_{day}_secret_key_2025"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

def minutes_since_update(time_str):
    if not time_str:
        return None
    try:
        now = now_saudi()
        last = now.replace(
            hour=int(time_str.split(":")[0]),
            minute=int(time_str.split(":")[1]),
            second=0, microsecond=0
        )
        diff = (now - last).total_seconds() / 60
        return diff
    except:
        return None

def is_driver_online(driver_name, shifts_data):
    driver_shifts = shifts_data.get(driver_name, [])
    if not driver_shifts:
        return True, None

    now = now_saudi()
    current_minutes = now.hour * 60 + now.minute

    for shift in driver_shifts:
        start_str = shift.get("start", "")
        end_str   = shift.get("end", "")
        if not start_str or not end_str:
            continue
        try:
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            start_m = sh * 60 + sm
            end_m   = eh * 60 + em
            if start_m <= end_m:
                if start_m <= current_minutes <= end_m:
                    return True, f"{start_str} - {end_str}"
            else:
                if current_minutes >= start_m or current_minutes <= end_m:
                    return True, f"{start_str} - {end_str}"
        except:
            continue
    return False, None

def save_daily_report(day, data, drivers, shifts_data):
    reports = load_json(REPORTS_FILE, {})
    rows  = []
    total = 0
    for driver in drivers:
        entry   = data.get(driver, {})
        count   = entry.get("count", 0)
        history = entry.get("history", [])
        total  += count
        rows.append({
            "اسم المندوب":    driver,
            "عدد الطلبات":    count,
            "آخر تحديث":      entry.get("time", "-"),
            "عدد التحديثات":  len(history),
            "توقيتات التحديثات": " | ".join(history) if history else "-"
        })
    rows.append({
        "اسم المندوب": "الإجمالي",
        "عدد الطلبات": total,
        "آخر تحديث": "",
        "عدد التحديثات": "",
        "توقيتات التحديثات": ""
    })
    reports[day] = {
        "saved_at": now_saudi().strftime("%Y-%m-%d %H:%M"),
        "rows":  rows,
        "total": total
    }
    save_json(REPORTS_FILE, reports)

# ==================================================================================
# تحميل البيانات
# ==================================================================================
drivers_list   = load_json(DRIVERS_FILE, [
    "أحمد محمد", "محمود علي", "عمر حسن", "يوسف إبراهيم", "مصطفى عبدالله",
])
config         = load_json(CONFIG_FILE,  {"base_url": "https://keitadeliveryanalysis-6zvs3tjytsugs3yweiq2s6.streamlit.app"})
all_data       = load_json(DATA_FILE,    {})
shifts_data    = load_json(SHIFTS_FILE,  {})
locations_data = load_json(LOCATIONS_FILE, {})
today          = get_today()

if today not in all_data:
    all_data[today] = {}

reports   = load_json(REPORTS_FILE, {})
yesterday = (now_saudi() - timedelta(days=1)).strftime("%Y-%m-%d")
if yesterday in all_data and yesterday not in reports:
    save_daily_report(yesterday, all_data[yesterday], drivers_list, shifts_data)

# ==================================================================================
# تحديد نوع الصفحة
# ==================================================================================
query_params   = st.query_params
driver_token   = query_params.get("driver", None)
mode           = query_params.get("mode", "driver")
current_driver = None
if driver_token:
    for d in drivers_list:
        if get_driver_token(d, today) == driver_token:
            current_driver = d
            break

# ==================================================================================
# CSS
# ==================================================================================
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    * { font-family: 'Cairo', sans-serif !important; direction: rtl; }
    .block-container { padding-top: 1.5rem !important; }
    .driver-card {
        background: white; border-radius: 10px; padding: 0.75rem 1rem;
        border: 1px solid #e9ecef; margin-bottom: 8px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .driver-card.danger  { border: 2px solid #dc3545 !important; background: #fff5f5 !important; }
    .driver-card.warning { border: 2px solid #ffc107 !important; background: #fffdf0 !important; }
    .driver-card.offline { border: 2px solid #adb5bd !important; background: #f8f9fa !important; }
    .driver-name  { font-weight: 600; font-size: 0.95rem; }
    .driver-count { font-size: 1.4rem; font-weight: 700; color: #1D9E75; }
    .driver-count.zero { color: #adb5bd; }
    .alert-badge { font-size: 0.7rem; padding: 2px 7px; border-radius: 12px; font-weight: 700; margin-right: 4px; }
    .badge-danger   { background: #dc3545; color: white; }
    .badge-warning  { background: #ffc107; color: #333; }
    .badge-offline  { background: #6c757d; color: white; }
    .badge-online   { background: #1D9E75; color: white; }
    .link-box {
        background: #f1f3f5; border-radius: 8px; padding: 8px 12px;
        font-family: monospace; font-size: 0.8rem; word-break: break-all;
        color: #495057; border: 1px solid #dee2e6;
    }
    .stButton > button { border-radius: 8px; font-family: 'Cairo', sans-serif !important; }
    div[data-testid="stMetricValue"] { direction: ltr; }
    .success-banner {
        background: #d4edda; color: #155724; padding: 12px 16px;
        border-radius: 8px; font-weight: 600; text-align: center; margin: 1rem 0;
    }
    .stNumberInput > div > div > input { text-align: center; font-size: 1.15rem; font-weight: 700; }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 10px; padding: 12px 16px; border: 1px solid #e9ecef;
    }
    .report-card {
        background: white; border-radius: 10px; padding: 1rem 1.25rem;
        border: 1px solid #e9ecef; margin-bottom: 10px;
    }
    .report-date  { font-weight: 700; font-size: 1rem; color: #1D9E75; }
    .report-total { font-size: 0.85rem; color: #6c757d; }
    .shift-box {
        background: #f1f3f5; border-radius: 8px; padding: 6px 10px;
        font-size: 0.8rem; color: #495057; border: 1px solid #dee2e6; margin-top: 4px;
    }
    /* PWA install banner */
    .pwa-banner {
        background: linear-gradient(135deg, #1D9E75, #159060);
        color: white; padding: 14px 18px; border-radius: 12px;
        text-align: center; margin: 1rem 0; font-weight: 700;
        box-shadow: 0 4px 12px rgba(29,158,117,0.3);
    }
    .pwa-banner small { font-size: 0.8rem; opacity: 0.85; font-weight: 400; }
    /* Map container */
    .map-wrapper { border-radius: 12px; overflow: hidden; border: 2px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

# ==================================================================================
# ██  PWA — Service Worker + Manifest (يظهر في صفحة المندوب فقط)  ██
# ==================================================================================
PWA_SCRIPT = """
<link rel="manifest" href="data:application/json;charset=utf-8,%7B%22name%22%3A%22%D8%B7%D9%84%D8%A8%D8%A7%D8%AA%D9%8A%20%D8%A7%D9%84%D9%8A%D9%88%D9%85%22%2C%22short_name%22%3A%22%D9%85%D9%86%D8%AF%D9%88%D8%A8%22%2C%22start_url%22%3A%22.%22%2C%22display%22%3A%22standalone%22%2C%22background_color%22%3A%22%23ffffff%22%2C%22theme_color%22%3A%221D9E75%22%2C%22icons%22%3A%5B%7B%22src%22%3A%22https%3A%2F%2Fcdn-icons-png.flaticon.com%2F512%2F2641%2F2641457.png%22%2C%22sizes%22%3A%22512x512%22%2C%22type%22%3A%22image%2Fpng%22%7D%5D%7D">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="مندوب">
<link rel="apple-touch-icon" href="https://cdn-icons-png.flaticon.com/512/2641/2641457.png">
<meta name="theme-color" content="#1D9E75">
<script>
// تسجيل Service Worker للـ PWA
if ('serviceWorker' in navigator) {
    const swCode = `
        const CACHE_NAME = 'driver-app-v1';
        self.addEventListener('install', e => self.skipWaiting());
        self.addEventListener('activate', e => self.clients.claim());
        self.addEventListener('fetch', e => {
            e.respondWith(
                fetch(e.request).catch(() => caches.match(e.request))
            );
        });
    `;
    const blob = new Blob([swCode], {type: 'application/javascript'});
    const swUrl = URL.createObjectURL(blob);
    navigator.serviceWorker.register(swUrl).catch(() => {});
}

// زر Add to Home Screen
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    const btn = document.getElementById('pwa-install-btn');
    if (btn) {
        btn.style.display = 'block';
        btn.addEventListener('click', () => {
            deferredPrompt.prompt();
            deferredPrompt.userChoice.then(() => { deferredPrompt = null; btn.style.display='none'; });
        });
    }
});
</script>
"""

# ==================================================================================
# ██  صفحة المناديب المشتركة  ██
# ==================================================================================
if mode != "admin" and not driver_token:
    st.markdown("## 🚚 Enter your number of orders for today.")
    st.caption(f"📅 {now_saudi().strftime('%A %d/%m/%Y')}")
    st.info("Each driver enters their order count next to their name, then press **Save All** at the bottom.")
    st.markdown("---")

    h1, h2, h3 = st.columns([3, 2, 2])
    h1.markdown("**Driver Name**")
    h2.markdown("**Number of Orders**")
    h3.markdown("**Last Updated**")

    inputs = {}
    for driver in drivers_list:
        prev      = all_data[today].get(driver, {}).get("count", 0)
        last_time = all_data[today].get(driver, {}).get("time", None)
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.markdown(f"<div style='padding:10px 4px; font-weight:600; font-size:1rem;'>{driver}</div>", unsafe_allow_html=True)
        with col2:
            inputs[driver] = st.number_input(label=driver, min_value=0, max_value=999, value=prev, step=1, key=f"inp_{driver}", label_visibility="collapsed")
        with col3:
            if last_time:
                st.success(f"✓ {last_time}")
            else:
                st.caption("—")

    st.markdown("---")
    if st.button("💾 Save All", type="primary", use_container_width=True):
        now_time = now_saudi().strftime("%H:%M")
        for driver, count in inputs.items():
            if count > 0 or driver in all_data[today]:
                entry = all_data[today].get(driver, {"count": 0, "time": None, "history": []})
                history = entry.get("history", [])
                history.append(now_time)
                all_data[today][driver] = {"count": count, "time": now_time, "history": history}
        save_json(DATA_FILE, all_data)
        st.success("✅ Saved successfully! The manager can now see the data.")
        st.rerun()

    st.markdown("---")
    st.caption("🔐 [Admin Login](?mode=admin)")
    st.stop()

# ==================================================================================
# ██  صفحة المندوب الفردي (مع GPS + PWA)  ██
# ==================================================================================
if current_driver:
    # حقن PWA في الصفحة
    st.markdown(PWA_SCRIPT, unsafe_allow_html=True)

    online, shift_label = is_driver_online(current_driver, shifts_data)
    st.title(f"🚚 مرحباً، {current_driver}")
    st.caption(f"📅 {now_saudi().strftime('%A %d/%m/%Y')}")

    if not online:
        st.warning("⏸️ أنت حالياً **خارج الدوام** — خارج ساعات الشيفت.")
        st.stop()

    # ── بنر الإضافة لشاشة الهوم ──
    st.markdown("""
    <div class="pwa-banner">
        📱 احفظ التطبيق على شاشتك الرئيسية وافتحه بسهولة كل يوم!<br>
        <small>على الآيفون: اضغط مشاركة ← إضافة إلى الشاشة الرئيسية &nbsp;|&nbsp; أندرويد: القائمة ← تثبيت التطبيق</small>
        <br><button id="pwa-install-btn" style="display:none; margin-top:8px; background:white; color:#1D9E75; border:none; padding:6px 18px; border-radius:20px; font-weight:700; cursor:pointer;">📲 تثبيت الآن</button>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── JavaScript: رفع الموقع كل 60 ثانية ──
    base_url_for_gps = config.get("base_url", "")
    gps_js = f"""
    <script>
    (function() {{
        const DRIVER_NAME = "{current_driver}";
        const TODAY       = "{today}";

        function sendLocation(lat, lng, acc) {{
            // نبعت الموقع عبر query param تحديث — Streamlit مش بيدعم POST من JS
            // فبنستخدم Image beacon لـ silent GET request
            const url = "{base_url_for_gps}?driver={driver_token}&lat=" + lat + "&lng=" + lng + "&acc=" + acc;
            // نخزن في localStorage كـ fallback
            const loc = {{lat, lng, acc, ts: new Date().toISOString()}};
            localStorage.setItem('driver_location_{current_driver}', JSON.stringify(loc));

            // نبعت fetch لنفس الصفحة بالإحداثيات (Streamlit هيتجاهلها لكن نكمل)
            try {{
                fetch(url, {{mode:'no-cors'}}).catch(()=>{{}});
            }} catch(e) {{}}
        }}

        function watchLocation() {{
            if (!navigator.geolocation) return;
            navigator.geolocation.watchPosition(
                pos => sendLocation(
                    pos.coords.latitude.toFixed(6),
                    pos.coords.longitude.toFixed(6),
                    Math.round(pos.coords.accuracy)
                ),
                err => console.warn('GPS error:', err.message),
                {{enableHighAccuracy: true, maximumAge: 30000, timeout: 15000}}
            );
        }}

        // ابدأ فوراً
        watchLocation();

        // كل 60 ثانية اطلب تحديث
        setInterval(() => {{
            navigator.geolocation.getCurrentPosition(
                pos => sendLocation(
                    pos.coords.latitude.toFixed(6),
                    pos.coords.longitude.toFixed(6),
                    Math.round(pos.coords.accuracy)
                ),
                () => {{}}
            );
        }}, 60000);

        // اعرض الحالة للمستخدم
        navigator.geolocation.getCurrentPosition(
            pos => {{
                const el = document.getElementById('gps-status');
                if (el) el.innerHTML = '📍 <b>تم تفعيل الموقع</b> — موقعك يُرسل تلقائياً للمدير كل دقيقة';
                sendLocation(pos.coords.latitude.toFixed(6), pos.coords.longitude.toFixed(6), Math.round(pos.coords.accuracy));
            }},
            err => {{
                const el = document.getElementById('gps-status');
                if (el) el.innerHTML = '⚠️ لم يُسمح بالموقع — اسمح للمتصفح بالوصول إلى موقعك';
            }},
            {{enableHighAccuracy: true, timeout: 10000}}
        );
    }})();
    </script>
    <div id="gps-status" style="background:#f0fff8;border:1px solid #1D9E75;border-radius:8px;padding:8px 14px;font-size:0.85rem;color:#155724;margin:8px 0;">
        ⏳ جاري تفعيل GPS...
    </div>
    """
    st.markdown(gps_js, unsafe_allow_html=True)

    # ── التحقق من إحداثيات مُمررة في URL (من JS) ──
    lat_param = query_params.get("lat", None)
    lng_param = query_params.get("lng", None)
    if lat_param and lng_param:
        try:
            locations_data[current_driver] = {
                "lat":       float(lat_param),
                "lng":       float(lng_param),
                "acc":       query_params.get("acc", "?"),
                "updated_at": now_saudi().strftime("%H:%M:%S")
            }
            save_json(LOCATIONS_FILE, locations_data)
        except:
            pass

    prev_count = all_data[today].get(current_driver, {}).get("count", 0)
    st.subheader("📦 أدخل عدد طلباتك لليوم")
    count = st.number_input("عدد الطلبات", min_value=0, max_value=999, value=prev_count, step=1, label_visibility="collapsed")

    if st.button("✅ إرسال", use_container_width=True, type="primary"):
        now_time = now_saudi().strftime("%H:%M")
        entry    = all_data[today].get(current_driver, {"count": 0, "time": None, "history": []})
        history  = entry.get("history", [])
        history.append(now_time)
        all_data[today][current_driver] = {"count": count, "time": now_time, "history": history}
        save_json(DATA_FILE, all_data)
        st.markdown('<div class="success-banner">✓ تم الإرسال! المدير شايف طلباتك دلوقتي.</div>', unsafe_allow_html=True)

    if prev_count > 0:
        st.info(f"آخر قيمة مسجلة: **{prev_count}** طلب")

    history_list = all_data[today].get(current_driver, {}).get("history", [])
    if history_list:
        st.caption(f"🕐 تحديثاتك اليوم ({len(history_list)}): " + " · ".join(history_list))
    st.stop()

# ==================================================================================
# ██  توكن غلط  ██
# ==================================================================================
if driver_token and not current_driver:
    st.error("❌ هذا الرابط غير صالح أو انتهت صلاحيته. اطلب رابطاً جديداً من المدير.")
    st.stop()

# ==================================================================================
# ██  صفحة المدير  ██
# ==================================================================================
if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False

if not st.session_state["admin_logged_in"]:
    st.markdown("## 🔐 دخول المدير")
    pw = st.text_input("كلمة السر", type="password")
    if st.button("دخول", type="primary"):
        if pw == ADMIN_PASSWORD:
            st.session_state["admin_logged_in"] = True
            st.rerun()
        else:
            st.error("كلمة السر غلط!")
    st.stop()

# ==================================================================================
# الداشبورد
# ==================================================================================
st.title("🗂️ داشبورد المناديب")
st.caption(f"📅 {now_saudi().strftime('%A %d/%m/%Y %H:%M')}")

today_data     = all_data.get(today, {})
total_orders   = sum(v.get("count", 0) for v in today_data.values())
active_drivers = sum(1 for v in today_data.values() if v.get("count", 0) > 0)
avg_orders     = round(total_orders / active_drivers, 1) if active_drivers > 0 else 0
top_driver     = max(today_data, key=lambda d: today_data[d].get("count", 0), default="-") if today_data else "-"
top_val        = today_data.get(top_driver, {}).get("count", 0) if top_driver != "-" else 0

late_drivers = []
for d in drivers_list:
    online, _ = is_driver_online(d, shifts_data)
    if not online:
        continue
    entry = today_data.get(d, {})
    t     = entry.get("time", None)
    mins  = minutes_since_update(t)
    if t is None or (mins is not None and mins > 60):
        late_drivers.append(d)

if late_drivers:
    names = ", ".join([f"**{x}**" for x in late_drivers])
    st.error(f"🚨 تنبيه: {len(late_drivers)} مندوب لم يحدّث بياناته — {names}")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("📦 إجمالي الطلبات", total_orders)
with col2: st.metric("👤 مناديب بطلبات", f"{active_drivers}/{len(drivers_list)}")
with col3: st.metric("📊 متوسط/مندوب", avg_orders)
with col4: st.metric("🏆 الأعلى أداء", f"{top_driver.split()[0]} ({top_val})" if top_val > 0 else "-")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 الطلبات اليومية",
    "🗺️ خريطة المناديب",
    "✏️ إدارة المناديب",
    "🔗 روابط المناديب",
    "📁 الأرشيف",
    "📊 التقارير اليومية"
])

# ==================================================================================
# تاب 1: الطلبات اليومية
# ==================================================================================
with tab1:
    col_search, col_refresh = st.columns([3, 1])
    with col_search:
        search = st.text_input("🔍 ابحث عن مندوب", placeholder="اكتب اسم المندوب...")
    with col_refresh:
        if st.button("🔄 تحديث", use_container_width=True):
            st.rerun()

    filtered = [d for d in drivers_list if search.lower() in d.lower()] if search else drivers_list

    cols = st.columns(3)
    for i, driver in enumerate(filtered):
        entry    = today_data.get(driver, {})
        count    = entry.get("count", 0)
        time_str = entry.get("time", None)
        history  = entry.get("history", [])
        mins     = minutes_since_update(time_str)
        online, shift_label = is_driver_online(driver, shifts_data)

        if not online:
            card_class = "driver-card offline"
            badge      = '<span class="alert-badge badge-offline">⏸ Offline</span>'
            status     = "خارج الشيفت"
        elif time_str is None:
            card_class = "driver-card danger"
            badge      = '<span class="alert-badge badge-danger">⚠ لم يبلّغ</span>'
            status     = "⚪ لم يُبلّغ بعد"
        elif mins is not None and mins > 60:
            card_class = "driver-card danger"
            badge      = f'<span class="alert-badge badge-danger">🔴 منذ {round(mins)} د</span>'
            status     = f"⏰ {time_str}"
        elif mins is not None and mins > 30:
            card_class = "driver-card warning"
            badge      = f'<span class="alert-badge badge-warning">🟡 منذ {round(mins)} د</span>'
            status     = f"⏰ {time_str}"
        else:
            card_class = "driver-card"
            badge      = '<span class="alert-badge badge-online">✓ Online</span>'
            status     = f"⏰ {time_str}" if time_str else "⚪ لم يُبلّغ بعد"

        # تفاصيل التحديثات
        updates_info = ""
        if history:
            updates_info = f"تحديثات ({len(history)}): " + " · ".join(history)

        color = "#1D9E75" if count > 0 and online else "#adb5bd"

        with cols[i % 3]:
            st.markdown(f"""
            <div class="{card_class}">
                <div>
                    <div class="driver-name">{driver} {badge}</div>
                    <div style="font-size:0.78rem; color:#6c757d;">{status}</div>
                    <div style="font-size:0.72rem; color:#adb5bd;">{updates_info}</div>
                </div>
                <div class="driver-count {'zero' if count == 0 else ''}" style="color:{color}">
                    {count}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col_exp, col_save, col_reset = st.columns([2, 1, 1])
    with col_exp:
        rows = []
        for driver in drivers_list:
            entry   = today_data.get(driver, {})
            history = entry.get("history", [])
            rows.append({
                "اسم المندوب":         driver,
                "عدد الطلبات":         entry.get("count", 0),
                "آخر تحديث":           entry.get("time", "-"),
                "عدد التحديثات":       len(history),
                "توقيتات التحديثات":  " | ".join(history) if history else "-"
            })
        df_export = pd.DataFrame(rows)
        total_val = df_export["عدد الطلبات"].sum()
        df_export.loc[len(df_export)] = ["الإجمالي", total_val, "", "", ""]

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, index=False, sheet_name=today)
            wb  = writer.book
            ws  = writer.sheets[today]
            hfmt = wb.add_format({'bold': True, 'bg_color': '#1D9E75', 'font_color': 'white', 'border': 1, 'align': 'center'})
            for cn, cv in enumerate(df_export.columns.values):
                ws.write(0, cn, cv, hfmt)
            ws.set_column(0, 0, 22); ws.set_column(1, 5, 18)

        st.download_button(
            label="⬇️ تحميل تقرير Excel", data=output.getvalue(),
            file_name=f"مناديب_{today}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col_save:
        if st.button("💾 حفظ تقرير اليوم", use_container_width=True, type="secondary"):
            save_daily_report(today, today_data, drivers_list, shifts_data)
            st.success("✅ تم الحفظ!")

    with col_reset:
        if st.button("🗑️ تصفير اليوم", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_reset"):
                save_daily_report(today, today_data, drivers_list, shifts_data)
                all_data[today] = {}
                save_json(DATA_FILE, all_data)
                st.session_state["confirm_reset"] = False
                st.success("تم الحفظ والتصفير!")
                st.rerun()
            else:
                st.session_state["confirm_reset"] = True
                st.warning("اضغط مرة تانية للتأكيد")

# ==================================================================================
# تاب 2: خريطة المناديب اللايف
# ==================================================================================
with tab2:
    st.subheader("🗺️ خريطة المناديب — Live")
    st.caption("يتحدث كل دقيقة · الدوائر الخضراء = Online · الرمادية = Offline")

    col_map_refresh, col_map_info = st.columns([1, 3])
    with col_map_refresh:
        if st.button("🔄 تحديث الخريطة", use_container_width=True):
            st.rerun()
    with col_map_info:
        located = sum(1 for d in drivers_list if d in locations_data)
        st.info(f"📍 {located} / {len(drivers_list)} مندوب شارك موقعه")

    # بناء JSON لبيانات المناديب على الخريطة
    map_drivers = []
    for driver in drivers_list:
        loc    = locations_data.get(driver, None)
        online, _ = is_driver_online(driver, shifts_data)
        entry  = today_data.get(driver, {})
        count  = entry.get("count", 0)
        updated = entry.get("time", None)

        if loc:
            map_drivers.append({
                "name":    driver,
                "lat":     loc["lat"],
                "lng":     loc["lng"],
                "online":  online,
                "count":   count,
                "updated": updated or "لم يُبلّغ",
                "gps_time": loc.get("updated_at", ""),
                "acc":     loc.get("acc", "?")
            })
        else:
            map_drivers.append({
                "name":    driver,
                "lat":     None,
                "lng":     None,
                "online":  online,
                "count":   count,
                "updated": updated or "لم يُبلّغ",
                "gps_time": None,
                "acc":     None
            })

    map_json = json.dumps(map_drivers, ensure_ascii=False)

    # خريطة Leaflet.js
    map_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; padding:0; font-family: 'Cairo', sans-serif; }}
  #map {{ width:100%; height:520px; }}
  .driver-popup {{ min-width:160px; text-align:right; direction:rtl; }}
  .driver-popup h4 {{ margin:0 0 6px; color:#1D9E75; font-size:1rem; }}
  .driver-popup p  {{ margin:2px 0; font-size:0.82rem; color:#555; }}
  .badge-on  {{ background:#1D9E75; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem; }}
  .badge-off {{ background:#6c757d; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem; }}
  .no-loc-list {{ padding:12px; background:#f8f9fa; border-radius:8px; margin-top:10px; }}
  .no-loc-list h5 {{ margin:0 0 6px; color:#6c757d; }}
  .no-loc-list li {{ font-size:0.85rem; color:#888; list-style:disc; margin-right:18px; }}
  #legend {{
    position:absolute; bottom:24px; right:12px; z-index:1000;
    background:white; padding:10px 14px; border-radius:10px;
    box-shadow:0 2px 8px rgba(0,0,0,0.15); font-size:0.8rem; direction:rtl;
  }}
  #legend div {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; }}
  .dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="legend">
  <div><span class="dot" style="background:#1D9E75;"></span> Online</div>
  <div><span class="dot" style="background:#6c757d;"></span> Offline</div>
  <div><span class="dot" style="background:#dc3545; border:2px solid #a00;"></span> لم يُبلّغ</div>
</div>

<script>
const driversData = {map_json};

// تجميع المناديب عندهم موقع
const withLoc    = driversData.filter(d => d.lat !== null);
const withoutLoc = driversData.filter(d => d.lat === null);

// مركز الخريطة
let centerLat = 24.7136, centerLng = 46.6753; // الرياض افتراضي
if (withLoc.length > 0) {{
    centerLat = withLoc.reduce((s,d)=>s+d.lat,0) / withLoc.length;
    centerLng = withLoc.reduce((s,d)=>s+d.lng,0) / withLoc.length;
}}

const map = L.map('map').setView([centerLat, centerLng], withLoc.length > 0 ? 12 : 6);

L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '© OpenStreetMap',
    maxZoom: 19
}}).addTo(map);

// دوائر المناديب
withLoc.forEach(d => {{
    const color  = d.online ? '#1D9E75' : '#6c757d';
    const border = d.updated === 'لم يُبلّغ' ? '#dc3545' : color;

    const circle = L.circleMarker([d.lat, d.lng], {{
        radius: 18,
        fillColor: color,
        color: border,
        weight: d.updated === 'لم يُبلّغ' ? 3 : 2,
        opacity: 1,
        fillOpacity: 0.85
    }}).addTo(map);

    // اسم مختصر داخل الدائرة
    const shortName = d.name.split(' ')[0];
    const label = L.divIcon({{
        className: '',
        html: `<div style="color:white;font-weight:700;font-size:0.72rem;text-align:center;pointer-events:none;white-space:nowrap;">${{shortName}}</div>`,
        iconAnchor: [0, -8]
    }});
    L.marker([d.lat, d.lng], {{icon: label, interactive: false}}).addTo(map);

    const badge = d.online
        ? '<span class="badge-on">Online</span>'
        : '<span class="badge-off">Offline</span>';

    const gpsInfo = d.gps_time ? `<p>📡 GPS: ${{d.gps_time}} (±${{d.acc}}م)</p>` : '';

    circle.bindPopup(`
        <div class="driver-popup">
            <h4>🚚 ${{d.name}}</h4>
            <p>${{badge}}</p>
            <p>📦 طلبات: <b>${{d.count}}</b></p>
            <p>🕐 آخر تبليغ: ${{d.updated}}</p>
            ${{gpsInfo}}
        </div>
    `);
}});

// تلقائي: ضبط الخريطة على كل المناديب
if (withLoc.length > 1) {{
    const bounds = L.latLngBounds(withLoc.map(d => [d.lat, d.lng]));
    map.fitBounds(bounds, {{padding: [40, 40]}});
}}

// تحديث تلقائي كل 60 ثانية
setTimeout(() => location.reload(), 60000);
</script>

<!-- قائمة بدون موقع -->
<div id="no-loc" style="display:${{withoutLoc.length>0?'block':'none'}};padding:8px 12px;">
</div>
<script>
if (withoutLoc.length > 0) {{
    const div = document.getElementById('no-loc');
    div.innerHTML = `
        <div class="no-loc-list">
            <h5>⚠️ المناديب التالية لم تفعّل GPS بعد (${{withoutLoc.length}}):</h5>
            <ul>${{withoutLoc.map(d=>`<li>${{d.name}}</li>`).join('')}}</ul>
        </div>
    `;
}}
</script>
</body>
</html>
"""
    st.components.v1.html(map_html, height=600, scrolling=False)

    # ── جدول مواقع المناديب ──
    if locations_data:
        st.markdown("### 📍 آخر موقع مسجل لكل مندوب")
        loc_rows = []
        for driver in drivers_list:
            loc = locations_data.get(driver, None)
            if loc:
                loc_rows.append({
                    "المندوب":     driver,
                    "خط العرض":   loc.get("lat", "-"),
                    "خط الطول":   loc.get("lng", "-"),
                    "دقة (م)":    loc.get("acc", "-"),
                    "آخر تحديث":  loc.get("updated_at", "-")
                })
        if loc_rows:
            st.dataframe(pd.DataFrame(loc_rows), use_container_width=True, hide_index=True)


# ==================================================================================
# تاب 3: إدارة المناديب + الشيفتات
# ==================================================================================
with tab3:
    st.markdown("### ➕ إضافة مناديب جدد")
    st.info("اكتب كل اسم في سطر منفصل ثم اضغط إضافة")
    new_names_input = st.text_area("أسماء جديدة", placeholder="محمد إبراهيم\nعلي حسن", height=120, label_visibility="collapsed")
    if st.button("➕ إضافة", type="primary"):
        new_names = [n.strip() for n in new_names_input.splitlines() if n.strip()]
        added = 0
        for name in new_names:
            if name not in drivers_list:
                drivers_list.append(name)
                added += 1
        save_json(DRIVERS_FILE, drivers_list)
        st.success(f"✅ تم إضافة {added} مندوب!")
        st.rerun()

    st.markdown("---")
    st.markdown("### ✏️ تعديل / حذف / شيفتات المناديب")
    st.caption("اضغط على اسم المندوب لتعديل شيفتاته — أو اضغط 🗑️ لحذفه")

    if st.session_state.get("pending_delete"):
        driver_to_del = st.session_state["pending_delete"]
        st.warning(f"⚠️ هتحذف **{driver_to_del}** — اكتب كلمة سر المدير للتأكيد")
        col_pw, col_confirm, col_cancel = st.columns([3, 1, 1])
        with col_pw:
            del_pw = st.text_input("كلمة السر", type="password", key="del_pw_input", label_visibility="collapsed", placeholder="كلمة سر المدير")
        with col_confirm:
            if st.button("✅ تأكيد", type="primary", use_container_width=True):
                if del_pw == ADMIN_PASSWORD:
                    if driver_to_del in drivers_list:
                        drivers_list.remove(driver_to_del)
                    save_json(DRIVERS_FILE, drivers_list)
                    st.session_state["pending_delete"] = None
                    st.success(f"✅ تم حذف {driver_to_del}")
                    st.rerun()
                else:
                    st.error("❌ كلمة السر غلط!")
        with col_cancel:
            if st.button("إلغاء", use_container_width=True):
                st.session_state["pending_delete"] = None
                st.rerun()
        st.markdown("---")

    for i, driver in enumerate(drivers_list):
        safe_key = driver.replace(" ", "_").replace("/", "_")
        driver_shifts = shifts_data.get(driver, [])
        online, shift_lbl = is_driver_online(driver, shifts_data)

        with st.expander(f"{'🟢' if online else '⚫'} {driver}  —  {'Online' if online else 'Offline'}", expanded=False):
            c_edit, c_del = st.columns([5, 1])
            with c_edit:
                new_name = st.text_input("الاسم", value=driver, key=f"edit_{safe_key}", label_visibility="collapsed")
                drivers_list[i] = new_name
            with c_del:
                if st.button("🗑️", key=f"del_{safe_key}", help=f"حذف {driver}"):
                    st.session_state["pending_delete"] = driver
                    st.rerun()

            st.markdown("**⏰ الشيفتات (حتى 3 شيفتات):**")
            updated_shifts = []
            for si in range(3):
                shift = driver_shifts[si] if si < len(driver_shifts) else {"start": "", "end": ""}
                sc1, sc2, sc3 = st.columns([2, 2, 1])
                with sc1:
                    s_start = st.text_input(f"بداية شيفت {si+1}", value=shift.get("start", ""), placeholder="08:00", key=f"shift_start_{safe_key}_{si}")
                with sc2:
                    s_end   = st.text_input(f"نهاية شيفت {si+1}", value=shift.get("end", ""), placeholder="16:00", key=f"shift_end_{safe_key}_{si}")
                with sc3:
                    st.markdown("<div style='padding-top:28px; font-size:0.8rem; color:#adb5bd;'>HH:MM</div>", unsafe_allow_html=True)
                if s_start and s_end:
                    updated_shifts.append({"start": s_start.strip(), "end": s_end.strip()})

            if st.button(f"💾 حفظ شيفتات {driver}", key=f"save_shift_{safe_key}", type="secondary"):
                shifts_data[driver] = updated_shifts
                save_json(SHIFTS_FILE, shifts_data)
                st.success("✅ تم حفظ الشيفتات!")
                st.rerun()

            if driver_shifts:
                shifts_text = "  |  ".join([f"شيفت {j+1}: {s['start']} ← {s['end']}" for j, s in enumerate(driver_shifts)])
                st.markdown(f'<div class="shift-box">📋 {shifts_text}</div>', unsafe_allow_html=True)
            else:
                st.caption("⚠️ لا توجد شيفتات — المندوب يُعتبر Online دائماً")

    st.markdown("---")
    if st.button("💾 حفظ تعديلات الأسماء", type="secondary", use_container_width=True):
        save_json(DRIVERS_FILE, drivers_list)
        st.success("✅ تم الحفظ!")

# ==================================================================================
# تاب 4: روابط المناديب
# ==================================================================================
with tab4:
    st.info("⚠️ الروابط دي بتتغير كل يوم تلقائياً — ابعتها للمناديب كل صباح على واتساب.")
    saved_url = config.get("base_url", "https://keitadeliveryanalysis-6zvs3tjytsugs3yweiq2s6.streamlit.app")
    base_url  = st.text_input("🌐 رابط الموقع الأساسي", value=saved_url)
    if st.button("💾 حفظ الرابط", type="secondary"):
        config["base_url"] = base_url
        save_json(CONFIG_FILE, config)
        st.success("✅ تم حفظ الرابط!")

    st.markdown("---")
    st.markdown("### روابط اليوم لكل مندوب")
    for driver in drivers_list:
        token     = get_driver_token(driver, today)
        full_link = f"{base_url}?driver={token}"
        col_name, col_link = st.columns([2, 5])
        with col_name: st.write(f"**{driver}**")
        with col_link: st.markdown(f'<div class="link-box">{full_link}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### رسالة واتساب جاهزة")
    wa_msg = f"🚚 *تقرير الطلبات - {now_saudi().strftime('%d/%m/%Y')}*\n\nكل مندوب يفتح رابطه:\n\n"
    for driver in drivers_list:
        token   = get_driver_token(driver, today)
        wa_msg += f"▫️ {driver}: {base_url}?driver={token}\n"
    wa_msg += "\n💡 افتح الرابط واضغط 'إضافة للشاشة الرئيسية' عشان تلاقيه بسرعة كل يوم"
    st.text_area("رسالة واتساب", wa_msg, height=300)

# ==================================================================================
# تاب 5: الأرشيف
# ==================================================================================
with tab5:
    st.subheader("📁 سجل الأيام السابقة")
    available_dates = sorted(all_data.keys(), reverse=True)
    if not available_dates:
        st.info("مفيش بيانات محفوظة لحد دلوقتي.")
    else:
        selected_date = st.selectbox("اختار يوم", available_dates)
        day_data = all_data.get(selected_date, {})
        rows = []
        for driver in drivers_list:
            entry   = day_data.get(driver, {})
            history = entry.get("history", [])
            rows.append({
                "اسم المندوب":        driver,
                "عدد الطلبات":        entry.get("count", 0),
                "وقت آخر تحديث":      entry.get("time", "-"),
                "عدد التحديثات":      len(history),
                "توقيتات التحديثات": " | ".join(history) if history else "-"
            })
        df_archive = pd.DataFrame(rows)
        st.dataframe(df_archive, use_container_width=True, hide_index=True)

        output2 = BytesIO()
        with pd.ExcelWriter(output2, engine="xlsxwriter") as writer:
            df_archive.to_excel(writer, index=False, sheet_name=selected_date)
            wb   = writer.book
            ws   = writer.sheets[selected_date]
            hfmt = wb.add_format({'bold': True, 'bg_color': '#1D9E75', 'font_color': 'white', 'border': 1, 'align': 'center'})
            for cn, cv in enumerate(df_archive.columns.values):
                ws.write(0, cn, cv, hfmt)
            ws.set_column(0, 0, 22); ws.set_column(1, 5, 18)
        st.download_button(
            label=f"⬇️ تحميل تقرير {selected_date}", data=output2.getvalue(),
            file_name=f"مناديب_{selected_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==================================================================================
# تاب 6: التقارير اليومية المحفوظة
# ==================================================================================
with tab6:
    st.subheader("📊 التقارير اليومية المحفوظة")
    st.caption("بتتحفظ تلقائياً كل يوم — أو يدوياً من تاب الطلبات")

    saved_reports = load_json(REPORTS_FILE, {})
    report_dates  = sorted(saved_reports.keys(), reverse=True)

    if not report_dates:
        st.info("مفيش تقارير محفوظة لحد دلوقتي.")
    else:
        st.markdown("### ملخص الأيام")
        n_cols = min(len(report_dates), 4)
        summary_cols = st.columns(n_cols)
        for idx, rdate in enumerate(report_dates[:4]):
            rep = saved_reports[rdate]
            with summary_cols[idx % n_cols]:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-date">📅 {rdate}</div>
                    <div class="report-total">إجمالي: <b>{rep.get('total', 0)}</b> طلب</div>
                    <div class="report-total" style="font-size:0.75rem;color:#adb5bd;">حُفظ: {rep.get('saved_at','-')}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        selected_report = st.selectbox("اختار يوم لعرض تقريره", report_dates, key="sel_report")
        rep_data = saved_reports[selected_report]
        rep_rows = rep_data.get("rows", [])

        if rep_rows:
            df_rep = pd.DataFrame(rep_rows)
            # عرض عمود التوقيتات بشكل واضح
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

            # إحصائيات التحديثات
            data_rows = [r for r in rep_rows if r.get("اسم المندوب") != "الإجمالي"]
            if data_rows:
                st.markdown("### 📈 إحصائيات التحديثات")
                stats_cols = st.columns(3)
                total_updates = sum(int(r.get("عدد التحديثات", 0)) for r in data_rows if str(r.get("عدد التحديثات","")).isdigit())
                max_updates_row = max(data_rows, key=lambda r: int(r.get("عدد التحديثات", 0)) if str(r.get("عدد التحديثات","")).isdigit() else 0)
                with stats_cols[0]:
                    st.metric("مجموع كل التحديثات", total_updates)
                with stats_cols[1]:
                    st.metric("أكثر مندوب تحديثاً", max_updates_row.get("اسم المندوب", "-"))
                with stats_cols[2]:
                    st.metric("عدد تحديثاته", max_updates_row.get("عدد التحديثات", 0))

            out_rep = BytesIO()
            with pd.ExcelWriter(out_rep, engine="xlsxwriter") as writer:
                df_rep.to_excel(writer, index=False, sheet_name=selected_report)
                wb   = writer.book
                ws   = writer.sheets[selected_report]
                hfmt = wb.add_format({'bold': True, 'bg_color': '#1D9E75', 'font_color': 'white', 'border': 1, 'align': 'center'})
                for cn, cv in enumerate(df_rep.columns.values):
                    ws.write(0, cn, cv, hfmt)
                ws.set_column(0, 0, 22); ws.set_column(1, 5, 18)

            st.download_button(
                label=f"⬇️ تحميل تقرير {selected_report}", data=out_rep.getvalue(),
                file_name=f"تقرير_{selected_report}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.markdown("---")
        if st.button("🗑️ حذف هذا التقرير", type="secondary"):
            if st.session_state.get("confirm_del_report"):
                del saved_reports[selected_report]
                save_json(REPORTS_FILE, saved_reports)
                st.session_state["confirm_del_report"] = False
                st.success("تم الحذف!")
                st.rerun()
            else:
                st.session_state["confirm_del_report"] = True
                st.warning("اضغط مرة تانية للتأكيد")

st.markdown("---")
if st.button("🚪 تسجيل خروج"):
    st.session_state["admin_logged_in"] = False
    st.rerun()
