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
LOCATIONS_FILE = "locations.json"   # ← مواقع المناديب اللايف

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
            "عدد التحديثات":  len(history),           # ← الجديد
            "أوقات التحديثات": " | ".join(history) if history else "-"  # ← الجديد
        })
    rows.append({
        "اسم المندوب": "الإجمالي", "عدد الطلبات": total,
        "آخر تحديث": "", "عدد التحديثات": "", "أوقات التحديثات": ""
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
    .loc-banner {
        background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 10px;
        padding: 10px 14px; font-size: 0.85rem; color: #2e7d32; margin: 8px 0;
    }
    .loc-banner.warn {
        background: #fff8e1; border-color: #ffe082; color: #f57f17;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================================
# JavaScript: تتبع الموقع (يُستخدم في صفحة المندوب)
# ==================================================================================
LOCATION_JS = """
<script>
// ========= إعدادات =========
const DRIVER_NAME   = "{driver_name}";
const UPDATE_URL    = "{base_url}?driver={token}&mode=location_update";
const INTERVAL_MS   = 60000;   // 60 ثانية
const SW_PATH       = "/app/static/sw.js";   // مش مستخدم هنا — fallback فقط

// ========= إرسال الموقع =========
function sendLocation(lat, lng, acc) {{
    const payload = {{
        driver: DRIVER_NAME,
        lat: lat,
        lng: lng,
        acc: acc,
        ts:  new Date().toISOString()
    }};
    // نرسل للـ Streamlit عن طريق localStorage كـ trigger
    // (Streamlit مش بيدعم POST من JS مباشرة — بنستخدم حيلة الـ anchor)
    const key = "loc_" + DRIVER_NAME.replace(/ /g,"_");
    localStorage.setItem(key, JSON.stringify(payload));

    // كمان نرسل لـ server-side endpoint (لو موجود)
    fetch("{update_endpoint}", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(payload),
        keepalive: true
    }}).catch(()=>{{}});
}}

// ========= مراقبة الموقع =========
if ("geolocation" in navigator) {{
    const watchId = navigator.geolocation.watchPosition(
        pos => sendLocation(
            pos.coords.latitude,
            pos.coords.longitude,
            pos.coords.accuracy
        ),
        err => console.warn("GPS error:", err.message),
        {{ enableHighAccuracy: true, maximumAge: 30000, timeout: 20000 }}
    );

    // احتفظ بالـ watchId في localStorage عشان ما يتوقفش
    localStorage.setItem("gps_watch", watchId);
}}

// ========= Wake Lock (يمنع الشاشة من الإغلاق) =========
async function keepAwake() {{
    if ("wakeLock" in navigator) {{
        try {{
            await navigator.wakeLock.request("screen");
            console.log("Wake lock active");
        }} catch(e) {{
            console.warn("Wake lock failed:", e);
        }}
    }}
}}
keepAwake();

// ========= تحديث دوري كل دقيقة حتى لو مفيش حركة =========
setInterval(() => {{
    navigator.geolocation.getCurrentPosition(
        pos => sendLocation(
            pos.coords.latitude,
            pos.coords.longitude,
            pos.coords.accuracy
        ),
        err => {{}}
    );
}}, INTERVAL_MS);
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
# ██  صفحة المندوب الفردي  ██
# ==================================================================================
if current_driver:
    online, shift_label = is_driver_online(current_driver, shifts_data)
    base_url_cfg = config.get("base_url", "")
    token_val    = get_driver_token(current_driver, today)

    st.title(f"🚚 Hello, {current_driver}")
    st.caption(f"📅 Today: {now_saudi().strftime('%A %d/%m/%Y')}")

    if not online:
        st.warning("⏸️ You are currently **Offline** — outside your shift hours.")
        st.stop()

    # ── إدراج JavaScript تتبع الموقع ──
    update_endpoint = f"{base_url_cfg}?driver={token_val}&mode=location_update"
    js_code = LOCATION_JS.format(
        driver_name     = current_driver,
        base_url        = base_url_cfg,
        token           = token_val,
        update_endpoint = update_endpoint
    )
    st.components.v1.html(js_code, height=0)

    # ── تعليمات الموقع ──
    loc_entry = locations_data.get(current_driver, {})
    if loc_entry.get("lat"):
        last_loc_time = loc_entry.get("ts", "")[:16].replace("T", " ")
        st.markdown(f"""
        <div class="loc-banner">
            📍 موقعك وصل للمدير — آخر تحديث: {last_loc_time}
            (lat: {round(loc_entry['lat'],4)}, lng: {round(loc_entry['lng'],4)})
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="loc-banner warn">
            📍 سيطلب التطبيق إذن الموقع — اضغط "السماح" عشان المدير يشوف مكانك على الخريطة
        </div>""", unsafe_allow_html=True)

    # ── استقبال الموقع من localStorage عبر JS callback ──
    # (Streamlit لا يدعم POST من JS مباشرة، فبنستخدم st.query_params كحل بديل)
    raw_lat = query_params.get("lat", None)
    raw_lng = query_params.get("lng", None)
    if raw_lat and raw_lng:
        try:
            loc_data = {
                "lat": float(raw_lat),
                "lng": float(raw_lng),
                "ts":  now_saudi().isoformat()
            }
            locations_data[current_driver] = loc_data
            save_json(LOCATIONS_FILE, locations_data)
        except:
            pass

    st.markdown("---")
    prev_count = all_data[today].get(current_driver, {}).get("count", 0)
    st.subheader("📦 Enter your number of orders for today.")
    count = st.number_input("Number of Orders", min_value=0, max_value=999, value=prev_count, step=1, label_visibility="collapsed")

    if st.button("✅ Submit", use_container_width=True, type="primary"):
        now_time = now_saudi().strftime("%H:%M")
        entry    = all_data[today].get(current_driver, {"count": 0, "time": None, "history": []})
        history  = entry.get("history", [])
        history.append(now_time)
        all_data[today][current_driver] = {"count": count, "time": now_time, "history": history}
        save_json(DATA_FILE, all_data)
        st.markdown('<div class="success-banner">✓ Submitted successfully! The manager can see your orders now.</div>', unsafe_allow_html=True)

    if prev_count > 0:
        st.info(f"Last recorded value: **{prev_count}** orders")

    history_list = all_data[today].get(current_driver, {}).get("history", [])
    if history_list:
        st.caption(f"🕐 Your updates today ({len(history_list)}): " + " · ".join(history_list))

    # ── زر "شارك موقعك" بديل احتياطي ──
    st.markdown("---")
    st.markdown("#### 📍 مشاركة موقعك يدوياً (احتياطي)")
    st.caption("اضغط الزر دا لو الموقع مش بيتبعت تلقائياً")
    share_js = f"""
    <button onclick="
        navigator.geolocation.getCurrentPosition(function(p){{
            window.location = '?driver={token_val}&lat='+p.coords.latitude+'&lng='+p.coords.longitude;
        }}, function(){{ alert('تعذّر الحصول على موقعك — تأكد من إعطاء الإذن'); }},
        {{enableHighAccuracy:true, timeout:10000}});
    " style="
        background:#1D9E75; color:white; border:none; border-radius:8px;
        padding:10px 20px; font-size:1rem; font-family:Cairo,sans-serif;
        cursor:pointer; width:100%;
    ">📍 أرسل موقعي الآن</button>
    """
    st.components.v1.html(share_js, height=60)

    st.stop()

# ==================================================================================
# ██  توكن غلط  ██
# ==================================================================================
if driver_token and not current_driver:
    st.error("❌ This link is invalid or has expired. Please request a new link from the manager.")
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
    "📋 الطلبات اليومية", "✏️ إدارة المناديب",
    "🔗 روابط المناديب", "📁 الأرشيف",
    "📊 التقارير اليومية", "🗺️ خريطة المناديب"
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

        updates_count = len(history)
        updates_info  = f"🔁 تحديثات اليوم: {updates_count}" if history else "🔁 لم يُحدّث بعد"
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
                "اسم المندوب":    driver,
                "عدد الطلبات":    entry.get("count", 0),
                "آخر تحديث":      entry.get("time", "-"),
                "عدد التحديثات":  len(history),
                "أوقات التحديثات": " | ".join(history) if history else "-"
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
            ws.set_column(0, 0, 22); ws.set_column(1, 4, 18)

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
# تاب 2: إدارة المناديب + الشيفتات
# ==================================================================================
with tab2:
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

            st.markdown("**⏰ الشيفتات:**")
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
# تاب 3: روابط المناديب
# ==================================================================================
with tab3:
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
    st.text_area("رسالة واتساب", wa_msg, height=280)

# ==================================================================================
# تاب 4: الأرشيف
# ==================================================================================
with tab4:
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
                "اسم المندوب":    driver,
                "عدد الطلبات":    entry.get("count", 0),
                "وقت التحديث":    entry.get("time", "-"),
                "عدد التحديثات":  len(history),
                "أوقات التحديثات": " | ".join(history) if history else "-"
            })
        df_archive = pd.DataFrame(rows)
        st.dataframe(df_archive, use_container_width=True, hide_index=True)

        output2 = BytesIO()
        with pd.ExcelWriter(output2, engine="xlsxwriter") as writer:
            df_archive.to_excel(writer, index=False, sheet_name=selected_date)
        st.download_button(
            label=f"⬇️ تحميل تقرير {selected_date}", data=output2.getvalue(),
            file_name=f"مناديب_{selected_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==================================================================================
# تاب 5: التقارير اليومية المحفوظة
# ==================================================================================
with tab5:
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
            # ── عرض عدد التحديثات وأوقاتها بشكل بارز ──
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

            # ── ملخص التحديثات لكل مندوب ──
            st.markdown("#### 🔁 تفاصيل التحديثات")
            for row in rep_rows:
                if row.get("اسم المندوب") == "الإجمالي":
                    continue
                n_updates = row.get("عدد التحديثات", 0)
                times_str = row.get("أوقات التحديثات", "-")
                if n_updates and int(str(n_updates)) > 0:
                    st.markdown(f"**{row['اسم المندوب']}** — {n_updates} تحديث: `{times_str}`")
                else:
                    st.markdown(f"**{row['اسم المندوب']}** — لم يُحدّث")

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

# ==================================================================================
# تاب 6: خريطة المناديب اللايف  🗺️
# ==================================================================================
with tab6:
    st.subheader("🗺️ خريطة المناديب — مواقع لايف")
    st.caption("بيتحدث كل مرة المندوب يفتح الرابط أو يضغط 'أرسل موقعي'")

    col_map_refresh, col_map_info = st.columns([1, 3])
    with col_map_refresh:
        if st.button("🔄 تحديث الخريطة", use_container_width=True):
            locations_data = load_json(LOCATIONS_FILE, {})
            st.rerun()
    with col_map_info:
        active_locs = sum(1 for d in drivers_list if locations_data.get(d, {}).get("lat"))
        st.info(f"📍 {active_locs} من {len(drivers_list)} مناديب مواقعهم ظاهرة")

    # ── بناء بيانات الخريطة ──
    map_drivers = []
    for driver in drivers_list:
        loc = locations_data.get(driver, {})
        if loc.get("lat") and loc.get("lng"):
            online, _ = is_driver_online(driver, shifts_data)
            orders    = today_data.get(driver, {}).get("count", 0)
            updates   = len(today_data.get(driver, {}).get("history", []))
            last_ts   = loc.get("ts", "")[:16].replace("T", " ")
            map_drivers.append({
                "name":    driver,
                "lat":     loc["lat"],
                "lng":     loc["lng"],
                "online":  online,
                "orders":  orders,
                "updates": updates,
                "ts":      last_ts,
            })

    if not map_drivers:
        st.warning("⚠️ مفيش مناديب شاركوا موقعهم لحد دلوقتي.\n\nالمناديب لازم يفتحوا الرابط بتاعهم ويسمحوا للتطبيق ياخد موقعهم.")
    else:
        # ── خريطة Leaflet عبر HTML ──
        markers_js = ""
        for d in map_drivers:
            color = "#1D9E75" if d["online"] else "#adb5bd"
            popup = f"{d['name']}<br>📦 {d['orders']} طلب | 🔁 {d['updates']} تحديث<br>🕐 {d['ts']}"
            markers_js += f"""
            L.circleMarker([{d['lat']}, {d['lng']}], {{
                radius: 14, color: '{color}', fillColor: '{color}',
                fillOpacity: 0.85, weight: 3
            }}).addTo(map)
            .bindPopup(`{popup}`)
            .bindTooltip(`{d['name']}`, {{permanent: true, direction: 'top', offset: [0, -14],
                className: 'driver-tooltip'}});
            """

        center_lat = sum(d["lat"] for d in map_drivers) / len(map_drivers)
        center_lng = sum(d["lng"] for d in map_drivers) / len(map_drivers)

        map_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  body {{ margin:0; padding:0; }}
  #map {{ height: 480px; width: 100%; border-radius: 12px; }}
  .driver-tooltip {{
    background: white; border: 2px solid #1D9E75; border-radius: 6px;
    font-family: Cairo, sans-serif; font-weight: 700; font-size: 12px;
    color: #1D9E75; padding: 2px 6px; white-space: nowrap;
  }}
  .leaflet-popup-content {{ font-family: Cairo, sans-serif; font-size: 13px;
    direction: rtl; text-align: right; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
var map = L.map('map').setView([{center_lat}, {center_lng}], 12);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '© OpenStreetMap', maxZoom: 19
}}).addTo(map);
{markers_js}
</script>
</body>
</html>
"""
        st.components.v1.html(map_html, height=500)

        # ── جدول المواقع ──
        st.markdown("#### 📋 تفاصيل المواقع")
        loc_rows = []
        for d in map_drivers:
            loc_rows.append({
                "المندوب":     d["name"],
                "الحالة":      "🟢 Online" if d["online"] else "⚫ Offline",
                "الطلبات":     d["orders"],
                "التحديثات":   d["updates"],
                "آخر موقع":    d["ts"],
                "lat":         round(d["lat"], 5),
                "lng":         round(d["lng"], 5),
            })
        st.dataframe(pd.DataFrame(loc_rows), use_container_width=True, hide_index=True)

    # ── معلومات للمدير عن آلية التتبع ──
    with st.expander("ℹ️ كيف يشتغل التتبع؟"):
        st.markdown("""
**الآلية:**
- لما المندوب يفتح الرابط بتاعه، التطبيق بيطلب إذن الموقع
- لو وافق، الموقع بيتبعت تلقائياً كل دقيقة طول ما الصفحة مفتوحة
- لو أغلق المتصفح، التتبع بيوقف (ده قيد من المتصفح نفسه)
- عنده زر "أرسل موقعي الآن" لو حاب يبعت يدوياً

**نصيحة:**
- قول للمناديب يفتحوا الرابط في أول الشيفت ويخلوا التبويب مفتوح
- على Android: في Chrome، ممكن يضيف الموقع للشاشة الرئيسية كـ PWA يشتغل في الخلفية
        """)

st.markdown("---")
if st.button("🚪 تسجيل خروج"):
    st.session_state["admin_logged_in"] = False
    st.rerun()
