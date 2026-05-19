import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timezone, timedelta
from io import BytesIO

SAUDI_TZ = timezone(timedelta(hours=3))
def now_saudi():
    return datetime.now(SAUDI_TZ)

st.set_page_config(page_title="تتبع طلبات المناديب", page_icon="🚚", layout="wide", initial_sidebar_state="collapsed")

DATA_FILE    = "delivery_data.json"
DRIVERS_FILE = "drivers.json"
CONFIG_FILE  = "config.json"
REPORTS_FILE = "daily_reports.json"
SHIFTS_FILE  = "shifts.json"
ADMIN_PASSWORD = "admin123"

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

# ── تعديل 1: توكن ثابت بناءً على اسم المندوب بس (بدون تاريخ) ──
def get_driver_token(driver_name):
    import hashlib
    return hashlib.md5(f"{driver_name}_fixed_secret_key_2025".encode()).hexdigest()[:10]

def minutes_since_update(time_str):
    if not time_str:
        return None
    try:
        now = now_saudi()
        last = now.replace(hour=int(time_str.split(":")[0]), minute=int(time_str.split(":")[1]), second=0, microsecond=0)
        return (now - last).total_seconds() / 60
    except:
        return None

def is_driver_online(driver_name, shifts_data):
    driver_shifts = shifts_data.get(driver_name, [])
    if not driver_shifts:
        return True, None
    now = now_saudi()
    cur = now.hour * 60 + now.minute
    for shift in driver_shifts:
        try:
            sh, sm = map(int, shift["start"].split(":"))
            eh, em = map(int, shift["end"].split(":"))
            s, e = sh*60+sm, eh*60+em
            if (s <= e and s <= cur <= e) or (s > e and (cur >= s or cur <= e)):
                return True, f"{shift['start']} - {shift['end']}"
        except:
            continue
    return False, None

def calc_idle_periods(loc_hist, threshold=30):
    if len(loc_hist) < 2:
        return []
    idle_periods, idle_start, idle_start_time = [], None, None
    for i in range(1, len(loc_hist)):
        prev, curr = loc_hist[i-1], loc_hist[i]
        try:
            moved = abs(float(curr["lat"])-float(prev["lat"])) > 0.0005 or abs(float(curr["lng"])-float(prev["lng"])) > 0.0005
        except:
            moved = True
        if not moved:
            if idle_start is None:
                idle_start, idle_start_time = i-1, prev.get("time","")
        else:
            if idle_start is not None:
                try:
                    sh,sm = map(int,idle_start_time.split(":")); eh,em = map(int,prev.get("time","0:0").split(":"))
                    diff = (eh*60+em)-(sh*60+sm)
                except:
                    diff = 0
                if diff >= threshold:
                    idle_periods.append({"from":idle_start_time,"to":prev.get("time",""),"minutes":diff,"lat":loc_hist[idle_start]["lat"],"lng":loc_hist[idle_start]["lng"]})
                idle_start = None
    if idle_start is not None:
        try:
            sh,sm = map(int,idle_start_time.split(":")); eh,em = map(int,loc_hist[-1].get("time","0:0").split(":"))
            diff = (eh*60+em)-(sh*60+sm)
        except:
            diff = 0
        if diff >= threshold:
            idle_periods.append({"from":idle_start_time,"to":loc_hist[-1].get("time",""),"minutes":diff,"lat":loc_hist[idle_start]["lat"],"lng":loc_hist[idle_start]["lng"],"still_idle":True})
    return idle_periods

def save_daily_report(day, data, drivers, shifts_data):
    reports = load_json(REPORTS_FILE, {})
    rows, total = [], 0
    for driver in drivers:
        entry = data.get(driver, {})
        count = entry.get("count", 0)
        history = entry.get("history", [])
        idle_periods = calc_idle_periods(entry.get("location_history", []))
        idle_summary = " | ".join([f"{p['from']}→{p['to']}({p['minutes']}د)" for p in idle_periods]) if idle_periods else "-"
        total += count
        rows.append({"اسم المندوب":driver,"عدد الطلبات":count,"آخر تحديث":entry.get("time","-"),
                     "عدد التحديثات":len(history),"سجل التحديثات":" | ".join(history) if history else "-","فترات التوقف":idle_summary})
    rows.append({"اسم المندوب":"الإجمالي","عدد الطلبات":total,"آخر تحديث":"","عدد التحديثات":"","سجل التحديثات":"","فترات التوقف":""})
    reports[day] = {"saved_at":now_saudi().strftime("%Y-%m-%d %H:%M"),"rows":rows,"total":total}
    save_json(REPORTS_FILE, reports)

# تحميل البيانات
drivers_list = load_json(DRIVERS_FILE, ["أحمد محمد","محمود علي","عمر حسن","يوسف إبراهيم","مصطفى عبدالله"])
config       = load_json(CONFIG_FILE,  {"base_url":"https://keitadeliveryanalysis-6zvs3tjytsugs3yweiq2s6.streamlit.app"})
all_data     = load_json(DATA_FILE,    {})
shifts_data  = load_json(SHIFTS_FILE,  {})
today        = get_today()

if today not in all_data:
    all_data[today] = {}

reports   = load_json(REPORTS_FILE, {})
yesterday = (now_saudi()-timedelta(days=1)).strftime("%Y-%m-%d")
if yesterday in all_data and yesterday not in reports:
    save_daily_report(yesterday, all_data[yesterday], drivers_list, shifts_data)

query_params   = st.query_params
driver_token   = query_params.get("driver", None)
mode           = query_params.get("mode", "driver")
url_lat        = query_params.get("lat", None)
url_lng        = query_params.get("lng", None)

current_driver = None
if driver_token:
    for d in drivers_list:
        # ── تعديل 1: مقارنة بالتوكن الثابت بدون تاريخ ──
        if get_driver_token(d) == driver_token:
            current_driver = d
            break

# حفظ الموقع لو جه في الـ URL
if current_driver and url_lat and url_lng:
    try:
        entry = all_data[today].get(current_driver, {"count":0,"time":None,"history":[],"location_history":[]})
        loc_history = entry.get("location_history", [])
        now_time = now_saudi().strftime("%H:%M")
        last_loc = loc_history[-1] if loc_history else {}
        if not (last_loc.get("lat") == url_lat and last_loc.get("time","") == now_time):
            loc_history.append({"lat":url_lat,"lng":url_lng,"time":now_time})
            loc_history = loc_history[-200:]
            entry["location_history"] = loc_history
            entry["lat"] = url_lat
            entry["lng"] = url_lng
            all_data[today][current_driver] = entry
            save_json(DATA_FILE, all_data)
    except:
        pass

st.markdown("""
<style>
    header[data-testid="stHeader"]{display:none!important}
    #MainMenu{visibility:hidden!important}
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    *{font-family:'Cairo',sans-serif!important;direction:rtl}
    .block-container{padding-top:1.5rem!important}
    .driver-card{background:white;border-radius:10px;padding:.75rem 1rem;border:1px solid #e9ecef;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
    .driver-card.danger{border:2px solid #dc3545!important;background:#fff5f5!important}
    .driver-card.warning{border:2px solid #ffc107!important;background:#fffdf0!important}
    .driver-card.offline{border:2px solid #adb5bd!important;background:#f8f9fa!important}
    .driver-name{font-weight:600;font-size:.95rem}
    .driver-count{font-size:1.4rem;font-weight:700;color:#1D9E75}
    .driver-count.zero{color:#adb5bd}
    .alert-badge{font-size:.7rem;padding:2px 7px;border-radius:12px;font-weight:700;margin-right:4px}
    .badge-danger{background:#dc3545;color:white}
    .badge-warning{background:#ffc107;color:#333}
    .badge-offline{background:#6c757d;color:white}
    .badge-online{background:#1D9E75;color:white}
    .badge-idle{background:#fd7e14;color:white}
    .link-box{background:#f1f3f5;border-radius:8px;padding:8px 12px;font-family:monospace;font-size:.8rem;word-break:break-all;color:#495057;border:1px solid #dee2e6}
    .stButton>button{border-radius:8px}
    div[data-testid="stMetricValue"]{direction:ltr}
    .success-banner{background:#d4edda;color:#155724;padding:12px 16px;border-radius:8px;font-weight:600;text-align:center;margin:1rem 0}
    .stNumberInput>div>div>input{text-align:center;font-size:1.15rem;font-weight:700}
    div[data-testid="stMetric"]{background:#f8f9fa;border-radius:10px;padding:12px 16px;border:1px solid #e9ecef}
    .report-card{background:white;border-radius:10px;padding:1rem 1.25rem;border:1px solid #e9ecef;margin-bottom:10px}
    .report-date{font-weight:700;font-size:1rem;color:#1D9E75}
    .report-total{font-size:.85rem;color:#6c757d}
    .shift-box{background:#f1f3f5;border-radius:8px;padding:6px 10px;font-size:.8rem;color:#495057;border:1px solid #dee2e6;margin-top:4px}
    .loc-card{background:#e8f8f0;border:1.5px solid #1D9E75;border-radius:10px;padding:12px 16px;margin:10px 0;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

# ==================================================================================
# صفحة المناديب المشتركة
# ==================================================================================
if mode != "admin" and not driver_token:
    st.markdown("## 🚚 إدخال الطلبات اليومية")
    st.caption(f"📅 {now_saudi().strftime('%A %d/%m/%Y')}")
    st.info("كل مندوب يكتب عدد طلباته جنب اسمه ويضغط **حفظ الكل**")
    st.markdown("---")

    h1,h2,h3 = st.columns([3,2,2])
    h1.markdown("**اسم المندوب**"); h2.markdown("**عدد الطلبات**"); h3.markdown("**آخر تحديث**")

    inputs = {}
    for driver in drivers_list:
        prev      = all_data[today].get(driver,{}).get("count",0)
        last_time = all_data[today].get(driver,{}).get("time",None)
        c1,c2,c3 = st.columns([3,2,2])
        with c1: st.markdown(f"<div style='padding:10px 4px;font-weight:600;font-size:1rem;'>{driver}</div>",unsafe_allow_html=True)
        with c2: inputs[driver] = st.number_input(label=driver,min_value=0,max_value=999,value=prev,step=1,key=f"inp_{driver}",label_visibility="collapsed")
        with c3:
            if last_time: st.success(f"✓ {last_time}")
            else: st.caption("—")

    st.markdown("---")
    if st.button("💾 حفظ الكل", type="primary", use_container_width=True):
        now_time = now_saudi().strftime("%H:%M")
        for driver, count in inputs.items():
            if count > 0 or driver in all_data[today]:
                entry = all_data[today].get(driver,{"count":0,"time":None,"history":[],"location_history":[]})
                history = entry.get("history",[])
                history.append(now_time)
                all_data[today][driver] = {**entry,"count":count,"time":now_time,"history":history}
        save_json(DATA_FILE, all_data)
        st.success("✅ تم الحفظ!")
        st.rerun()

    st.markdown("---")
    st.caption("🔐 [دخول المدير](?mode=admin)")
    st.stop()

# ==================================================================================
# صفحة المندوب الفردي
# ==================================================================================
if current_driver:
    online, shift_label = is_driver_online(current_driver, shifts_data)

    if not online:
        st.title(f"⏸️ {current_driver}")
        st.warning("You are currently OFF SHIFT — not your working hours.")
        st.stop()

    st.title(f"🚚 Hello, {current_driver}!")
    st.caption(f"📅 {now_saudi().strftime('%A %d/%m/%Y')}  |  🕐 {now_saudi().strftime('%H:%M')}")

    prev_entry = all_data[today].get(current_driver,{})
    prev_count = prev_entry.get("count",0)
    prev_lat   = prev_entry.get("lat","")
    prev_lng   = prev_entry.get("lng","")

    base_url_clean   = config.get("base_url","").rstrip("/")
    # ── تعديل 1: توكن ثابت بدون تاريخ ──
    token_for_driver = get_driver_token(current_driver)

    # ── تعديل 2: JS بيطلب الموقع مرة واحدة بس (بيحفظ في localStorage إن الإذن اتأخد) ──
    # لو الإذن اتأخد قبل كده هيفضل يبعت الموقع تلقائي بدون ما يطلب مرة تانية
    loc_html = f"""
<div id="loc_status" style="background:#e8f8f0;border:1.5px solid #1D9E75;border-radius:10px;padding:12px 16px;margin:10px 0;font-size:.9rem;color:#155724;">
    ⏳ Checking location...
</div>
<div id="loc_denied" style="display:none;background:#fff3cd;border:1px solid #ffc107;border-radius:10px;padding:12px 16px;margin:10px 0;font-size:.85rem;color:#856404;">
    ⚠️ <b>Location access denied.</b><br><br>
    <b>iPhone (Safari):</b> Settings → Safari → Location → Allow<br>
    <b>Android (Chrome):</b> Tap the 🔒 lock icon → Permissions → Location → Allow<br><br>
    Then <b>refresh the page</b>.
</div>
<script>
var locationSent = false;
var LOC_KEY = 'loc_granted_{token_for_driver}';

function sendLocation(lat, lng, acc) {{
    document.getElementById('loc_status').style.display = 'block';
    document.getElementById('loc_denied').style.display = 'none';
    document.getElementById('loc_status').innerHTML =
        '✅ Location active — accuracy: ' + acc + 'm | 📍 ' + lat + ', ' + lng;

    if (!locationSent) {{
        locationSent = true;
        var newUrl = '{base_url_clean}/?driver={token_for_driver}&lat=' + lat + '&lng=' + lng;
        try {{
            window.parent.location.href = newUrl;
        }} catch(e) {{
            window.location.href = newUrl;
        }}
    }}
}}

function getLocation() {{
    if (!navigator.geolocation) {{
        document.getElementById('loc_status').innerHTML = '❌ Browser does not support GPS.';
        return;
    }}

    // لو الإذن اتأخد قبل كده: اطلب الموقع مباشرة بدون popup
    var alreadyGranted = localStorage.getItem(LOC_KEY) === 'yes';

    if (alreadyGranted) {{
        // اطلب الموقع بهدوء
        navigator.geolocation.getCurrentPosition(function(pos) {{
            var lat = pos.coords.latitude.toFixed(6);
            var lng = pos.coords.longitude.toFixed(6);
            var acc = Math.round(pos.coords.accuracy);
            sendLocation(lat, lng, acc);
        }}, function(err) {{
            // لو الإذن اتسحب
            if (err.code === 1) {{
                localStorage.removeItem(LOC_KEY);
                document.getElementById('loc_status').style.display = 'none';
                document.getElementById('loc_denied').style.display = 'block';
            }} else {{
                document.getElementById('loc_status').innerHTML = '⚠️ Could not get location — make sure GPS is on.';
            }}
        }}, {{enableHighAccuracy:true, timeout:15000, maximumAge:0}});
    }} else {{
        // أول مرة: اطلب الإذن
        navigator.geolocation.getCurrentPosition(function(pos) {{
            // حفظ إن الإذن اتأخد
            try {{ localStorage.setItem(LOC_KEY, 'yes'); }} catch(e) {{}}
            var lat = pos.coords.latitude.toFixed(6);
            var lng = pos.coords.longitude.toFixed(6);
            var acc = Math.round(pos.coords.accuracy);
            sendLocation(lat, lng, acc);
        }}, function(err) {{
            if (err.code === 1) {{
                document.getElementById('loc_status').style.display = 'none';
                document.getElementById('loc_denied').style.display = 'block';
            }} else if (err.code === 2) {{
                document.getElementById('loc_status').innerHTML = '⚠️ Could not get location — make sure GPS is on.';
            }} else {{
                document.getElementById('loc_status').innerHTML = '⏱ Timed out — retrying...';
                locationSent = false;
            }}
        }}, {{enableHighAccuracy:true, timeout:15000, maximumAge:0}});
    }}
}}

// شغّل فور ما الصفحة تفتح
getLocation();

// كل دقيقة يبعت الموقع تاني (بدون popup لأن الإذن اتأخد)
setInterval(function() {{
    locationSent = false;
    getLocation();
}}, 60000);
</script>
"""
    st.components.v1.html(loc_html, height=140)

    if prev_lat and prev_lng:
        st.caption(f"📍 Last saved location: {prev_lat}, {prev_lng}  —  {prev_entry.get('time','')}")

    st.markdown("---")
    st.subheader("📦 Enter your number of orders for today")
    count = st.number_input("Orders", min_value=0, max_value=999, value=prev_count, step=1, label_visibility="collapsed")

    if st.button("✅ Submit", use_container_width=True, type="primary"):
        now_time = now_saudi().strftime("%H:%M")
        entry    = all_data[today].get(current_driver,{"count":0,"time":None,"history":[],"location_history":[]})
        history  = entry.get("history",[])
        history.append(now_time)
        all_data[today][current_driver] = {**entry,"count":count,"time":now_time,"history":history}
        save_json(DATA_FILE, all_data)
        st.markdown('<div class="success-banner">✓ Submitted successfully! The manager can see your orders now.</div>', unsafe_allow_html=True)
        st.rerun()

    if prev_count > 0:
        st.info(f"Last recorded value: **{prev_count}** orders")

    history_list = all_data[today].get(current_driver,{}).get("history",[])
    if history_list:
        st.caption(f"🕐 Your updates today ({len(history_list)}): " + "  ·  ".join(history_list))

    st.markdown("---")
    st.markdown("""
    <div style='background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:10px 14px;font-size:.82rem;color:#856404;'>
    📌 <b>Keep this link handy:</b><br>
    Android: Browser menu ⋮ → <b>Add to Home Screen</b><br>
    iPhone: Safari → Share button → <b>Add to Home Screen</b>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

    st.stop()

# توكن غلط
if driver_token and not current_driver:
    st.error("❌ الرابط مش صحيح أو انتهى صلاحيته.")
    st.stop()

# ==================================================================================
# صفحة المدير
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

# الداشبورد
st.title("🗂️ داشبورد المناديب")
st.caption(f"📅 {now_saudi().strftime('%A %d/%m/%Y %H:%M')} (توقيت السعودية)")

today_data     = all_data.get(today,{})
total_orders   = sum(v.get("count",0) for v in today_data.values())
active_drivers = sum(1 for v in today_data.values() if v.get("count",0)>0)
avg_orders     = round(total_orders/active_drivers,1) if active_drivers>0 else 0
top_driver     = max(today_data,key=lambda d:today_data[d].get("count",0),default="-") if today_data else "-"
top_val        = today_data.get(top_driver,{}).get("count",0) if top_driver!="-" else 0

late_drivers,idle_alerts = [],[]
for d in drivers_list:
    online,_ = is_driver_online(d,shifts_data)
    if not online: continue
    entry = today_data.get(d,{})
    t = entry.get("time",None)
    mins = minutes_since_update(t)
    if t is None or (mins is not None and mins>60): late_drivers.append(d)
    for p in calc_idle_periods(entry.get("location_history",[])):
        if p.get("still_idle"): idle_alerts.append((d,p["minutes"],p["from"]))

if late_drivers:
    st.error(f"🚨 لم يحدّث: {', '.join(late_drivers)}")
for d,mins,since in idle_alerts:
    st.warning(f"⚠️ **{d}** واقف في نفس المكان منذ {since} ({mins} دقيقة)")

c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("📦 إجمالي الطلبات", total_orders)
with c2: st.metric("👤 مناديب بطلبات", f"{active_drivers}/{len(drivers_list)}")
with c3: st.metric("📊 متوسط/مندوب", avg_orders)
with c4: st.metric("🏆 الأعلى أداء", f"{top_driver.split()[0]} ({top_val})" if top_val>0 else "-")

st.markdown("---")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📋 الطلبات اليومية","🗺️ الخريطة اللايف",
    "✏️ إدارة المناديب","🔗 روابط المناديب",
    "📁 الأرشيف","📊 التقارير"
])

# ==================================================================================
# تاب 1: الطلبات اليومية
# ==================================================================================
with tab1:
    cs,cr = st.columns([3,1])
    with cs: search = st.text_input("🔍 ابحث عن مندوب", placeholder="اكتب اسم المندوب...")
    with cr:
        if st.button("🔄 تحديث", use_container_width=True): st.rerun()

    filtered = [d for d in drivers_list if search.lower() in d.lower()] if search else drivers_list
    cols = st.columns(3)
    for i,driver in enumerate(filtered):
        entry    = today_data.get(driver,{})
        count    = entry.get("count",0)
        time_str = entry.get("time",None)
        history  = entry.get("history",[])
        mins     = minutes_since_update(time_str)
        online,_ = is_driver_online(driver,shifts_data)
        idle_periods = calc_idle_periods(entry.get("location_history",[]))
        is_idle  = any(p.get("still_idle") for p in idle_periods)

        if not online:
            card_class,badge,status = "driver-card offline",'<span class="alert-badge badge-offline">⏸ Offline</span>',"خارج الشيفت"
        elif is_idle:
            card_class,badge = "driver-card danger",'<span class="alert-badge badge-idle">🟠 واقف</span>'
            status = f"⏰ {time_str}" if time_str else "⚪ لم يُبلّغ"
        elif time_str is None:
            card_class,badge,status = "driver-card danger",'<span class="alert-badge badge-danger">⚠ لم يبلّغ</span>',"⚪ لم يُبلّغ بعد"
        elif mins is not None and mins>60:
            card_class,badge = "driver-card danger",f'<span class="alert-badge badge-danger">🔴 منذ {round(mins)} د</span>'
            status = f"⏰ {time_str}"
        elif mins is not None and mins>30:
            card_class,badge = "driver-card warning",f'<span class="alert-badge badge-warning">🟡 منذ {round(mins)} د</span>'
            status = f"⏰ {time_str}"
        else:
            card_class,badge = "driver-card",'<span class="alert-badge badge-online">✓ Online</span>'
            status = f"⏰ {time_str}" if time_str else "⚪ لم يُبلّغ"

        color    = "#1D9E75" if count>0 and online else "#adb5bd"
        loc_icon = "📍" if entry.get("lat") else "📵"
        with cols[i%3]:
            st.markdown(f"""
            <div class="{card_class}">
                <div>
                    <div class="driver-name">{driver} {badge} {loc_icon}</div>
                    <div style="font-size:.78rem;color:#6c757d;">{status}</div>
                    <div style="font-size:.72rem;color:#adb5bd;">تحديثات اليوم: {len(history)}</div>
                </div>
                <div class="driver-count {'zero' if count==0 else ''}" style="color:{color}">{count}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    ce,cs2,cr2 = st.columns([2,1,1])
    with ce:
        rows = []
        for driver in drivers_list:
            entry = today_data.get(driver,{})
            history = entry.get("history",[])
            idle_periods = calc_idle_periods(entry.get("location_history",[]))
            idle_s = " | ".join([f"{p['from']}→{p['to']}({p['minutes']}د)" for p in idle_periods]) if idle_periods else "-"
            rows.append({"اسم المندوب":driver,"عدد الطلبات":entry.get("count",0),"آخر تحديث":entry.get("time","-"),
                         "عدد التحديثات":len(history),"سجل التحديثات":" | ".join(history) if history else "-","فترات التوقف":idle_s})
        df_export = pd.DataFrame(rows)
        df_export.loc[len(df_export)] = ["الإجمالي",df_export["عدد الطلبات"].sum(),"","","",""]
        output = BytesIO()
        with pd.ExcelWriter(output,engine="xlsxwriter") as writer:
            df_export.to_excel(writer,index=False,sheet_name=today)
            wb=writer.book; ws=writer.sheets[today]
            hfmt=wb.add_format({'bold':True,'bg_color':'#1D9E75','font_color':'white','border':1,'align':'center'})
            for cn,cv in enumerate(df_export.columns.values): ws.write(0,cn,cv,hfmt)
            ws.set_column(0,0,22); ws.set_column(1,5,18)
        st.download_button("⬇️ تحميل Excel",data=output.getvalue(),file_name=f"مناديب_{today}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    with cs2:
        if st.button("💾 حفظ تقرير اليوم",use_container_width=True,type="secondary"):
            save_daily_report(today,today_data,drivers_list,shifts_data)
            st.success("✅ تم!")
    with cr2:
        if st.button("🗑️ تصفير اليوم",use_container_width=True,type="secondary"):
            if st.session_state.get("confirm_reset"):
                save_daily_report(today,today_data,drivers_list,shifts_data)
                all_data[today]={}
                save_json(DATA_FILE,all_data)
                st.session_state["confirm_reset"]=False
                st.rerun()
            else:
                st.session_state["confirm_reset"]=True
                st.warning("اضغط مرة تانية للتأكيد")

# ==================================================================================
# تاب 2: الخريطة اللايف
# ==================================================================================
with tab2:
    cr,ci = st.columns([1,3])
    with cr:
        if st.button("🔄 تحديث الخريطة",use_container_width=True): st.rerun()
    with ci:
        active_locs = sum(1 for d in drivers_list if today_data.get(d,{}).get("lat"))
        st.info(f"📍 {active_locs} مندوب أرسل موقعه النهارده — الخريطة تتحدث تلقائياً كل دقيقة من تليفون المندوب")

    drivers_with_loc = []
    for driver in drivers_list:
        entry = today_data.get(driver,{})
        lat   = entry.get("lat")
        lng   = entry.get("lng")
        if lat and lng:
            try:
                online,_ = is_driver_online(driver,shifts_data)
                idle_periods = calc_idle_periods(entry.get("location_history",[]))
                is_idle  = any(p.get("still_idle") for p in idle_periods)
                idle_min = max([p["minutes"] for p in idle_periods if p.get("still_idle")],default=0)
                loc_time = entry.get("location_history",[{}])[-1].get("time",entry.get("time","-"))
                drivers_with_loc.append({
                    "name":driver,"lat":float(lat),"lng":float(lng),
                    "count":entry.get("count",0),"time":entry.get("time","-"),
                    "loc_time":loc_time,"online":online,"idle":is_idle,"idle_min":idle_min,
                    "updates":len(entry.get("history",[]))
                })
            except:
                pass

    if not drivers_with_loc:
        st.warning("📵 مفيش مناديب بعتوا موقعهم لحد دلوقتي.")
        st.info("💡 المندوب لازم يفتح رابطه الخاص ويسمح بالـ GPS — الموقع هيتبعت تلقائياً كل دقيقة.")
    else:
        center_lat = sum(d["lat"] for d in drivers_with_loc) / len(drivers_with_loc)
        center_lng = sum(d["lng"] for d in drivers_with_loc) / len(drivers_with_loc)

        markers_js = ""
        for d in drivers_with_loc:
            if not d["online"]:    color,label = "#6c757d","Offline"
            elif d["idle"]:        color,label = "#fd7e14",f"واقف {d['idle_min']}د"
            else:                  color,label = "#1D9E75","Online"

            idle_html = f"<br><b style='color:#fd7e14'>⚠️ واقف منذ {d['idle_min']} دقيقة</b>" if d["idle"] else ""
            safe = d['name'].replace("'","\\'")
            markers_js += f"""
L.circleMarker([{d['lat']},{d['lng']}],{{
    radius:18,color:'white',weight:3,fillColor:'{color}',fillOpacity:.95
}}).addTo(map)
.bindPopup(`<div style='font-family:Cairo,Arial;direction:rtl;text-align:right;min-width:170px;padding:4px'>
    <b style='font-size:1rem'>🚚 {safe}</b><br>
    📦 طلبات: <b>{d['count']}</b><br>
    🕐 آخر تحديث: {d['time']}<br>
    📍 آخر موقع: {d['loc_time']}
    {idle_html}
    <br><span style='background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:.75rem'>{label}</span>
</div>`,{{maxWidth:220}})
.bindTooltip('<b style="font-family:Cairo,Arial">{safe}</b><br><small>{d["count"]} طلب</small>',
    {{permanent:true,direction:'top',offset:[0,-22],opacity:.9}});
"""

        map_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
body{{margin:0;padding:0}}
#map{{width:100%;height:520px;border-radius:12px}}
.leaflet-tooltip{{background:rgba(0,0,0,.78);color:white;border:none;border-radius:8px;font-size:12px;padding:4px 10px;box-shadow:none}}
.leaflet-tooltip::before{{display:none}}
.legend-box{{background:white;padding:10px 14px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);font-family:Cairo,Arial;font-size:13px;line-height:26px}}
</style></head>
<body><div id="map"></div>
<script>
var map = L.map('map').setView([{center_lat},{center_lng}],12);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'© OpenStreetMap'}}).addTo(map);
{markers_js}
var leg=L.control({{position:'bottomright'}});
leg.onAdd=function(){{
    var d=L.DomUtil.create('div','legend-box');
    d.innerHTML='<b>المناديب</b><br>'+
        '<span style="color:#1D9E75;font-size:20px">●</span> Online<br>'+
        '<span style="color:#fd7e14;font-size:20px">●</span> واقف<br>'+
        '<span style="color:#6c757d;font-size:20px">●</span> Offline';
    return d;
}};
leg.addTo(map);
</script></body></html>"""

        st.components.v1.html(map_html, height=540)

        st.markdown("### 📋 تفاصيل المواقع")
        map_rows = []
        for d in drivers_with_loc:
            status = "⏸ Offline" if not d["online"] else ("🟠 واقف" if d["idle"] else "🟢 Online")
            map_rows.append({"المندوب":d["name"],"الحالة":status,"طلبات":d["count"],
                             "آخر تحديث طلبات":d["time"],"آخر تحديث موقع":d["loc_time"],
                             "توقف (د)":d["idle_min"] if d["idle"] else 0,
                             "Lat":round(d["lat"],5),"Lng":round(d["lng"],5)})
        st.dataframe(pd.DataFrame(map_rows),use_container_width=True,hide_index=True)

# ==================================================================================
# تاب 3: إدارة المناديب
# ==================================================================================
with tab3:
    st.markdown("### ➕ إضافة مناديب جدد")
    new_names_input = st.text_area("أسماء جديدة",placeholder="محمد إبراهيم\nعلي حسن",height=100,label_visibility="collapsed")
    if st.button("➕ إضافة",type="primary"):
        new_names=[n.strip() for n in new_names_input.splitlines() if n.strip()]
        added=0
        for name in new_names:
            if name not in drivers_list:
                drivers_list.append(name); added+=1
        save_json(DRIVERS_FILE,drivers_list)
        st.success(f"✅ تم إضافة {added} مندوب!")
        st.rerun()

    st.markdown("---")
    st.markdown("### ✏️ تعديل / حذف / شيفتات")

    if st.session_state.get("pending_delete"):
        dtd = st.session_state["pending_delete"]
        st.warning(f"⚠️ هتحذف **{dtd}** — اكتب كلمة السر للتأكيد")
        cp,cc,ccl = st.columns([3,1,1])
        with cp: del_pw=st.text_input("كلمة السر",type="password",key="del_pw_input",label_visibility="collapsed")
        with cc:
            if st.button("✅ تأكيد",type="primary",use_container_width=True):
                if del_pw==ADMIN_PASSWORD:
                    if dtd in drivers_list: drivers_list.remove(dtd)
                    save_json(DRIVERS_FILE,drivers_list)
                    st.session_state["pending_delete"]=None
                    st.rerun()
                else: st.error("❌ كلمة السر غلط!")
        with ccl:
            if st.button("إلغاء",use_container_width=True):
                st.session_state["pending_delete"]=None; st.rerun()
        st.markdown("---")

    for i,driver in enumerate(drivers_list):
        safe_key = driver.replace(" ","_").replace("/","_")
        driver_shifts = shifts_data.get(driver,[])
        online,_ = is_driver_online(driver,shifts_data)
        with st.expander(f"{'🟢' if online else '⚫'} {driver}  —  {'Online' if online else 'Offline'}",expanded=False):
            ce2,cd = st.columns([5,1])
            with ce2:
                new_name=st.text_input("الاسم",value=driver,key=f"edit_{safe_key}",label_visibility="collapsed")
                drivers_list[i]=new_name
            with cd:
                if st.button("🗑️",key=f"del_{safe_key}"):
                    st.session_state["pending_delete"]=driver; st.rerun()

            st.markdown("**⏰ الشيفتات:** (اتركها فاضية = Online دايماً)")
            updated_shifts=[]
            for si in range(3):
                shift=driver_shifts[si] if si<len(driver_shifts) else {"start":"","end":""}
                sc1,sc2,sc3=st.columns([2,2,1])
                with sc1: s_start=st.text_input(f"بداية {si+1}",value=shift.get("start",""),placeholder="08:00",key=f"ss_{safe_key}_{si}")
                with sc2: s_end=st.text_input(f"نهاية {si+1}",value=shift.get("end",""),placeholder="16:00",key=f"se_{safe_key}_{si}")
                with sc3: st.markdown("<div style='padding-top:28px;font-size:.8rem;color:#adb5bd'>HH:MM</div>",unsafe_allow_html=True)
                if s_start and s_end: updated_shifts.append({"start":s_start.strip(),"end":s_end.strip()})
            if st.button(f"💾 حفظ شيفتات {driver}",key=f"save_shift_{safe_key}",type="secondary"):
                shifts_data[driver]=updated_shifts
                save_json(SHIFTS_FILE,shifts_data)
                st.success("✅ تم!"); st.rerun()
            if driver_shifts:
                shifts_text = "  |  ".join([f"شيفت {j+1}: {s['start']} ← {s['end']}" for j,s in enumerate(driver_shifts)])
                st.markdown(f'<div class="shift-box">📋 {shifts_text}</div>', unsafe_allow_html=True)
            else:
                st.caption("⚠️ لا توجد شيفتات — Online دائماً")

    st.markdown("---")
    if st.button("💾 حفظ تعديلات الأسماء",type="secondary",use_container_width=True):
        save_json(DRIVERS_FILE,drivers_list); st.success("✅ تم الحفظ!")

# ==================================================================================
# تاب 4: روابط المناديب
# ==================================================================================
with tab4:
    # ── تعديل 1: اللينك ثابت — مش بيتغير كل يوم ──
    st.success("✅ الروابط دي ثابتة ومش بتتغير — ابعتها مرة واحدة بس لكل مندوب.")
    saved_url = config.get("base_url","https://keitadeliveryanalysis-6zvs3tjytsugs3yweiq2s6.streamlit.app")
    base_url  = st.text_input("🌐 رابط الموقع الأساسي",value=saved_url)
    if st.button("💾 حفظ الرابط",type="secondary"):
        config["base_url"]=base_url
        save_json(CONFIG_FILE,config)
        st.success("✅ تم حفظ الرابط!")

    st.markdown("---")
    st.markdown("### روابط المناديب (ثابتة)")
    for driver in drivers_list:
        # ── تعديل 1: توكن ثابت بدون تاريخ ──
        token     = get_driver_token(driver)
        full_link = f"{base_url}?driver={token}"
        cn,cl = st.columns([2,5])
        with cn: st.write(f"**{driver}**")
        with cl: st.markdown(f'<div class="link-box">{full_link}</div>',unsafe_allow_html=True)

    st.markdown("---")
    wa_msg = f"🚚 *روابط المناديب — ثابتة يوميا*\n\nكل مندوب يفتح رابطه ويحطه على الهوم سكرين:\n\n"
    for driver in drivers_list:
        wa_msg += f"▫️ {driver}: {base_url}?driver={get_driver_token(driver)}\n"
    st.text_area("رسالة واتساب",wa_msg,height=260)

# ==================================================================================
# تاب 5: الأرشيف
# ==================================================================================
with tab5:
    st.subheader("📁 سجل الأيام السابقة")
    available_dates = sorted(all_data.keys(),reverse=True)
    if not available_dates:
        st.info("مفيش بيانات محفوظة.")
    else:
        selected_date = st.selectbox("اختار يوم",available_dates)
        day_data = all_data.get(selected_date,{})
        rows=[]
        for driver in drivers_list:
            entry=day_data.get(driver,{})
            history=entry.get("history",[])
            idle_periods=calc_idle_periods(entry.get("location_history",[]))
            idle_s=" | ".join([f"{p['from']}→{p['to']}({p['minutes']}د)" for p in idle_periods]) if idle_periods else "-"
            rows.append({"اسم المندوب":driver,"عدد الطلبات":entry.get("count",0),"وقت التحديث":entry.get("time","-"),
                         "عدد التحديثات":len(history),"سجل التحديثات":" | ".join(history) if history else "-","فترات التوقف":idle_s})
        df_archive=pd.DataFrame(rows)
        st.dataframe(df_archive,use_container_width=True,hide_index=True)
        output2=BytesIO()
        with pd.ExcelWriter(output2,engine="xlsxwriter") as writer:
            df_archive.to_excel(writer,index=False,sheet_name=selected_date)
        st.download_button(f"⬇️ تحميل تقرير {selected_date}",data=output2.getvalue(),
            file_name=f"مناديب_{selected_date}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==================================================================================
# تاب 6: التقارير
# ==================================================================================
with tab6:
    st.subheader("📊 التقارير اليومية المحفوظة")
    saved_reports=load_json(REPORTS_FILE,{})
    report_dates=sorted(saved_reports.keys(),reverse=True)
    if not report_dates:
        st.info("مفيش تقارير محفوظة.")
    else:
        n_cols=min(len(report_dates),4)
        scols=st.columns(n_cols)
        for idx,rdate in enumerate(report_dates[:4]):
            rep=saved_reports[rdate]
            with scols[idx%n_cols]:
                st.markdown(f'<div class="report-card"><div class="report-date">📅 {rdate}</div><div class="report-total">إجمالي: <b>{rep.get("total",0)}</b> طلب</div><div class="report-total" style="font-size:.75rem;color:#adb5bd">حُفظ: {rep.get("saved_at","-")}</div></div>',unsafe_allow_html=True)

        st.markdown("---")
        sel_rep=st.selectbox("اختار يوم لعرض تقريره",report_dates,key="sel_report")
        rep_rows=saved_reports[sel_rep].get("rows",[])
        if rep_rows:
            df_rep=pd.DataFrame(rep_rows)
            st.dataframe(df_rep,use_container_width=True,hide_index=True)
            out_rep=BytesIO()
            with pd.ExcelWriter(out_rep,engine="xlsxwriter") as writer:
                df_rep.to_excel(writer,index=False,sheet_name=sel_rep)
                wb=writer.book; ws=writer.sheets[sel_rep]
                hfmt=wb.add_format({'bold':True,'bg_color':'#1D9E75','font_color':'white','border':1,'align':'center'})
                for cn,cv in enumerate(df_rep.columns.values): ws.write(0,cn,cv,hfmt)
                ws.set_column(0,0,22); ws.set_column(1,6,20)
            st.download_button(f"⬇️ تحميل تقرير {sel_rep}",data=out_rep.getvalue(),
                file_name=f"تقرير_{sel_rep}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)

        st.markdown("---")
        if st.button("🗑️ حذف هذا التقرير",type="secondary"):
            if st.session_state.get("confirm_del_report"):
                del saved_reports[sel_rep]
                save_json(REPORTS_FILE,saved_reports)
                st.session_state["confirm_del_report"]=False
                st.rerun()
            else:
                st.session_state["confirm_del_report"]=True
                st.warning("اضغط مرة تانية للتأكيد")

st.markdown("---")
if st.button("🚪 تسجيل خروج"):
    st.session_state["admin_logged_in"]=False
    st.rerun()
