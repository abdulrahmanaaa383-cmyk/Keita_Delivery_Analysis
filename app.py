import streamlit as st
import pandas as pd
import json
import os
from datetime import date, datetime
from io import BytesIO

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
DATA_FILE    = "delivery_data.json"
DRIVERS_FILE = "drivers.json"

ADMIN_PASSWORD = "admin123"   # <-- غيّر كلمة السر هنا

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
    return date.today().strftime("%Y-%m-%d")

def get_driver_token(driver_name: str, day: str) -> str:
    import hashlib
    raw = f"{driver_name}_{day}_secret_key_2025"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

# ==================================================================================
# تحميل البيانات
# ==================================================================================

# ── قائمة المناديب: تُحفظ في drivers.json وتُدار من صفحة المدير ──
drivers_list = load_json(DRIVERS_FILE, [
    "أحمد محمد",
    "محمود علي",
    "عمر حسن",
    "يوسف إبراهيم",
    "مصطفى عبدالله",
])

# ── بيانات الطلبات اليومية ──
all_data = load_json(DATA_FILE, {})
today    = get_today()
if today not in all_data:
    all_data[today] = {}

# ==================================================================================
# تحديد نوع الصفحة
# ==================================================================================
query_params  = st.query_params
driver_token  = query_params.get("driver", None)
mode          = query_params.get("mode", "driver")

# هل التوكن يطابق مندوباً اليوم؟
current_driver = None
if driver_token:
    for d in drivers_list:
        if get_driver_token(d, today) == driver_token:
            current_driver = d
            break

# ==================================================================================
# CSS مشترك
# ==================================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    * { font-family: 'Cairo', sans-serif !important; direction: rtl; }
    .block-container { padding-top: 1.5rem !important; }
    .metric-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #1D9E75; }
    .metric-lbl { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
    .driver-card {
        background: white;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border: 1px solid #e9ecef;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .driver-name  { font-weight: 600; font-size: 0.95rem; }
    .driver-count { font-size: 1.4rem; font-weight: 700; color: #1D9E75; }
    .driver-count.zero { color: #adb5bd; }
    .link-box {
        background: #f1f3f5;
        border-radius: 8px;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 0.8rem;
        word-break: break-all;
        color: #495057;
        border: 1px solid #dee2e6;
    }
    .stButton > button  { border-radius: 8px; font-family: 'Cairo', sans-serif !important; }
    div[data-testid="stMetricValue"] { direction: ltr; }
    .success-banner {
        background: #d4edda;
        color: #155724;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    .stNumberInput > div > div > input {
        text-align: center;
        font-size: 1.15rem;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================================
# ██  صفحة المناديب — لينك واحد لكل المناديب  ██
# ==================================================================================
if mode != "admin" and not driver_token:

    st.markdown("## 🚚 إدخال الطلبات اليومية")
    st.caption(f"📅 {datetime.now().strftime('%A %d/%m/%Y')}")
    st.info("كل مندوب يكتب عدد طلباته جنب اسمه ويضغط **حفظ الكل** في الآخر")
    st.markdown("---")

    # رأس الجدول
    h1, h2, h3 = st.columns([3, 2, 2])
    h1.markdown("** driver name**")
    h2.markdown("**count orders**")
    h3.markdown("**last update**")

    inputs = {}
    for driver in drivers_list:
        prev      = all_data[today].get(driver, {}).get("count", 0)
        last_time = all_data[today].get(driver, {}).get("time", None)

        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.markdown(
                f"<div style='padding:10px 4px; font-weight:600; font-size:1rem;'>{driver}</div>",
                unsafe_allow_html=True
            )
        with col2:
            inputs[driver] = st.number_input(
                label=driver,
                min_value=0,
                max_value=999,
                value=prev,
                step=1,
                key=f"inp_{driver}",
                label_visibility="collapsed"
            )
        with col3:
            if last_time:
                st.success(f"✓ {last_time}")
            else:
                st.caption("—")

    st.markdown("---")

    if st.button("💾 حفظ الكل", type="primary", use_container_width=True):
        now_time = datetime.now().strftime("%H:%M")
        for driver, count in inputs.items():
            if count > 0 or driver in all_data[today]:
                all_data[today][driver] = {"count": count, "time": now_time}
        save_json(DATA_FILE, all_data)
        st.success("✅ تم الحفظ! البيانات ظهرت عند المدير دلوقتي.")
        st.rerun()

    st.markdown("---")
    st.caption("🔐 [دخول المدير](?mode=admin)")
    st.stop()

# ==================================================================================
# ██  صفحة المندوب الفردي (توكن يومي)  ██
# ==================================================================================
if current_driver:
    st.title(f"🚚 أهلاً، {current_driver}")
    st.caption(f"📅 اليوم: {datetime.now().strftime('%A %d/%m/%Y')}")
    st.markdown("---")

    prev_count = all_data[today].get(current_driver, {}).get("count", 0)

    st.subheader("📦 أدخل عدد طلباتك النهارده")
    count = st.number_input(
        "عدد الطلبات",
        min_value=0,
        max_value=999,
        value=prev_count,
        step=1,
        label_visibility="collapsed"
    )

    if st.button("✅ إرسال", use_container_width=True, type="primary"):
        all_data[today][current_driver] = {
            "count": count,
            "time": datetime.now().strftime("%H:%M")
        }
        save_json(DATA_FILE, all_data)
        st.markdown(
            '<div class="success-banner">✓ تم الإرسال بنجاح! المدير شايف طلباتك دلوقتي.</div>',
            unsafe_allow_html=True
        )

    if prev_count > 0:
        st.info(f"آخر قيمة مسجلة: **{prev_count}** طلب")

    st.stop()

# ==================================================================================
# ██  توكن غلط  ██
# ==================================================================================
if driver_token and not current_driver:
    st.error("❌ الرابط ده مش صحيح أو انتهى صلاحيته. طلب رابط جديد من المدير.")
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

# ---- الداشبورد ----
st.title("🗂️ داشبورد المناديب")
st.caption(f"📅 {datetime.now().strftime('%A %d/%m/%Y %H:%M')}")

today_data     = all_data.get(today, {})
total_orders   = sum(v.get("count", 0) for v in today_data.values())
active_drivers = sum(1 for v in today_data.values() if v.get("count", 0) > 0)
avg_orders     = round(total_orders / active_drivers, 1) if active_drivers > 0 else 0
top_driver     = max(today_data, key=lambda d: today_data[d].get("count", 0), default="-") if today_data else "-"
top_val        = today_data.get(top_driver, {}).get("count", 0) if top_driver != "-" else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 إجمالي الطلبات", total_orders)
with col2:
    st.metric("👤 مناديب بطلبات", f"{active_drivers}/{len(drivers_list)}")
with col3:
    st.metric("📊 متوسط/مندوب", avg_orders)
with col4:
    st.metric("🏆 الأعلى أداء", f"{top_driver.split()[0]} ({top_val})" if top_val > 0 else "-")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📋 الطلبات اليومية", "✏️ إدارة المناديب", "🔗 روابط المناديب", "📁 الأرشيف"])

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
        with cols[i % 3]:
            color  = "#1D9E75" if count > 0 else "#adb5bd"
            status = f"⏰ {time_str}" if time_str else "⚪ لم يُبلّغ بعد"
            st.markdown(f"""
            <div class="driver-card">
                <div>
                    <div class="driver-name">{driver}</div>
                    <div style="font-size:0.78rem; color:#6c757d;">{status}</div>
                </div>
                <div class="driver-count {'zero' if count == 0 else ''}" style="color:{color}">
                    {count}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_exp, col_reset = st.columns([2, 1])
    with col_exp:
        rows = []
        for driver in drivers_list:
            entry = today_data.get(driver, {})
            rows.append({
                "اسم المندوب": driver,
                "عدد الطلبات": entry.get("count", 0),
                "آخر تحديث":   entry.get("time", "-")
            })
        df_export = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["اسم المندوب", "عدد الطلبات", "آخر تحديث"])
        total_val = df_export["عدد الطلبات"].sum() if not df_export.empty else 0
        df_export.loc[len(df_export)] = ["الإجمالي", total_val, ""]

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, index=False, sheet_name=today)
            workbook  = writer.book
            worksheet = writer.sheets[today]
            header_fmt = workbook.add_format({
                'bold': True, 'bg_color': '#1D9E75', 'font_color': 'white',
                'border': 1, 'align': 'center'
            })
            for col_num, value in enumerate(df_export.columns.values):
                worksheet.write(0, col_num, value, header_fmt)
            worksheet.set_column(0, 0, 25)
            worksheet.set_column(1, 2, 18)

        st.download_button(
            label="⬇️ تحميل تقرير Excel",
            data=output.getvalue(),
            file_name=f"مناديب_{today}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col_reset:
        if st.button("🗑️ تصفير اليوم", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_reset"):
                all_data[today] = {}
                save_json(DATA_FILE, all_data)
                st.session_state["confirm_reset"] = False
                st.success("تم التصفير!")
                st.rerun()
            else:
                st.session_state["confirm_reset"] = True
                st.warning("اضغط مرة تانية للتأكيد")

# ==================================================================================
# تاب 2: إدارة المناديب  ← جديد
# ==================================================================================
with tab2:
    st.markdown("### ➕ إضافة مناديب جدد")
    st.info("اكتب كل اسم في سطر منفصل ثم اضغط إضافة")
    new_names_input = st.text_area(
        "أسماء جديدة",
        placeholder="محمد إبراهيم\nعلي حسن\nكريم سامي",
        height=140,
        label_visibility="collapsed"
    )
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
    st.markdown("### ✏️ تعديل أو حذف مندوب")
    st.caption("عدّل الاسم مباشرة ثم اضغط 'حفظ التعديلات' — أو اضغط 🗑️ لحذف مندوب")

    to_delete = []
    for i, driver in enumerate(drivers_list):
        c_num, c_edit, c_del = st.columns([1, 5, 1])
        with c_num:
            st.markdown(
                f"<div style='padding:10px 0; color:#adb5bd; font-size:0.85rem; text-align:center'>{i+1}</div>",
                unsafe_allow_html=True
            )
        with c_edit:
            new_name = st.text_input(
                f"driver_{i}",
                value=driver,
                key=f"edit_{i}",
                label_visibility="collapsed"
            )
            drivers_list[i] = new_name
        with c_del:
            if st.button("🗑️", key=f"del_{i}", help=f"حذف {driver}"):
                to_delete.append(driver)

    if to_delete:
        for d in to_delete:
            if d in drivers_list:
                drivers_list.remove(d)
        save_json(DRIVERS_FILE, drivers_list)
        st.rerun()

    if st.button("💾 حفظ التعديلات", type="secondary", use_container_width=True):
        save_json(DRIVERS_FILE, drivers_list)
        st.success("✅ تم الحفظ!")

# ==================================================================================
# تاب 3: روابط المناديب
# ==================================================================================
with tab3:
    st.info("⚠️ الروابط دي بتتغير كل يوم تلقائياً — ابعتها للمناديب كل صباح على واتساب.")

    base_url = st.text_input(
        "🌐 رابط الموقع الأساسي",
        value="https://your-app.streamlit.app",
        help="الرابط بتاع تطبيقك على Streamlit Cloud"
    )

    st.markdown("### روابط اليوم لكل المندوب")
    for driver in drivers_list:
        token     = get_driver_token(driver, today)
        full_link = f"{base_url}?driver={token}"
        col_name, col_link, col_copy = st.columns([2, 4, 1])
        with col_name:
            st.write(f"**{driver}**")
        with col_link:
            st.markdown(f'<div class="link-box">{full_link}</div>', unsafe_allow_html=True)
        with col_copy:
            st.code(token, language=None)

    st.markdown("---")
    st.markdown("### رسالة واتساب جاهزة (انسخها وابعتها)")
    wa_msg  = f"🚚 *تقرير الطلبات - {datetime.now().strftime('%d/%m/%Y')}*\n\n"
    wa_msg += "كل مندوب يفتح رابطه ويحدث عدد طلباته:\n\n"
    for driver in drivers_list:
        token   = get_driver_token(driver, today)
        wa_msg += f"▫️ {driver}: {base_url}?driver={token}\n"
    st.text_area("رسالة واتساب", wa_msg, height=300)

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
            entry = day_data.get(driver, {})
            rows.append({
                "اسم المندوب": driver,
                "عدد الطلبات": entry.get("count", 0),
                "وقت التحديث": entry.get("time", "-")
            })
        df_archive = pd.DataFrame(rows)
        st.dataframe(df_archive, use_container_width=True, hide_index=True)

        output2 = BytesIO()
        with pd.ExcelWriter(output2, engine="xlsxwriter") as writer:
            df_archive.to_excel(writer, index=False, sheet_name=selected_date)
        st.download_button(
            label=f"⬇️ تحميل تقرير {selected_date}",
            data=output2.getvalue(),
            file_name=f"مناديب_{selected_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
if st.button("🚪 تسجيل خروج"):
    st.session_state["admin_logged_in"] = False
    st.rerun()
