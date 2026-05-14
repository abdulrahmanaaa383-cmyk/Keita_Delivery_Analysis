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
# قائمة المناديب — عدّل الأسماء هنا
# ==================================================================================
DRIVERS = [
    "أحمد محمد",
    "محمود علي",
    "عمر حسن",
    "يوسف إبراهيم",
    "مصطفى عبدالله",
    # أضف باقي الأسماء هنا حتى 40 مندوب
    # "اسم المندوب 6",
    # "اسم المندوب 7",
    # ...
]

# ==================================================================================
# ملف حفظ البيانات
# ==================================================================================
DATA_FILE = "delivery_data.json"

def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_data(data: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_today() -> str:
    return date.today().strftime("%Y-%m-%d")

def get_driver_token(driver_name: str, day: str) -> str:
    """يولّد رمز يومي فريد لكل مندوب — يتغير كل يوم تلقائياً"""
    import hashlib
    raw = f"{driver_name}_{day}_secret_key_2025"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

# ==================================================================================
# تحميل البيانات
# ==================================================================================
all_data = load_data()
today = get_today()
if today not in all_data:
    all_data[today] = {}

# ==================================================================================
# تحديد نوع الصفحة (مدير أو مندوب)
# ==================================================================================
query_params = st.query_params
driver_token = query_params.get("driver", None)

# تحقق هل التوكن يطابق مندوباً اليوم
current_driver = None
if driver_token:
    for d in DRIVERS:
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
    .driver-name { font-weight: 600; font-size: 0.95rem; }
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
    .stButton > button {
        border-radius: 8px;
        font-family: 'Cairo', sans-serif !important;
    }
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
</style>
""", unsafe_allow_html=True)

# ==================================================================================
# صفحة المندوب
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
        save_data(all_data)
        st.markdown(
            '<div class="success-banner">✓ تم الإرسال بنجاح! المدير شايف طلباتك دلوقتي.</div>',
            unsafe_allow_html=True
        )

    if prev_count > 0:
        st.info(f"آخر قيمة مسجلة: **{prev_count}** طلب")

    st.stop()

# ==================================================================================
# صفحة غير معروفة (توكن خاطئ)
# ==================================================================================
if driver_token and not current_driver:
    st.error("❌ الرابط ده مش صحيح أو انتهى صلاحيته. طلب رابط جديد من المدير.")
    st.stop()

# ==================================================================================
# صفحة المدير (الداشبورد)
# ==================================================================================
st.title("🗂️ داشبورد المناديب")
st.caption(f"📅 {datetime.now().strftime('%A %d/%m/%Y %H:%M')}")

# ---- مقاييس سريعة ----
today_data = all_data.get(today, {})
total_orders = sum(v.get("count", 0) for v in today_data.values())
active_drivers = sum(1 for v in today_data.values() if v.get("count", 0) > 0)
avg_orders = round(total_orders / active_drivers, 1) if active_drivers > 0 else 0
top_driver = max(today_data, key=lambda d: today_data[d].get("count", 0), default="-")
top_val = today_data.get(top_driver, {}).get("count", 0)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 إجمالي الطلبات", total_orders)
with col2:
    st.metric("👤 مناديب بطلبات", f"{active_drivers}/{len(DRIVERS)}")
with col3:
    st.metric("📊 متوسط/مندوب", avg_orders)
with col4:
    st.metric("🏆 الأعلى أداء", f"{top_driver.split()[0]} ({top_val})" if top_val > 0 else "-")

st.markdown("---")

# ---- تبويبات ----
tab1, tab2, tab3 = st.tabs(["📋 الطلبات اليومية", "🔗 روابط المناديب", "📁 الأرشيف"])

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

    filtered = [d for d in DRIVERS if search.lower() in d.lower()] if search else DRIVERS

    # عرض كروت المناديب
    cols = st.columns(3)
    for i, driver in enumerate(filtered):
        entry = today_data.get(driver, {})
        count = entry.get("count", 0)
        time_str = entry.get("time", None)
        with cols[i % 3]:
            color = "#1D9E75" if count > 0 else "#adb5bd"
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

    # تصدير Excel
    col_exp, col_reset = st.columns([2, 1])
    with col_exp:
        rows = []
        for driver in DRIVERS:
            entry = today_data.get(driver, {})
            rows.append({
                "اسم المندوب": driver,
                "عدد الطلبات": entry.get("count", 0),
                "آخر تحديث": entry.get("time", "-")
            })
        df_export = pd.DataFrame(rows)
        df_export.loc[len(df_export)] = ["الإجمالي", df_export["عدد الطلبات"].sum(), ""]

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, index=False, sheet_name=today)
            workbook = writer.book
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
                save_data(all_data)
                st.session_state["confirm_reset"] = False
                st.success("تم التصفير!")
                st.rerun()
            else:
                st.session_state["confirm_reset"] = True
                st.warning("اضغط مرة تانية للتأكيد")

# ==================================================================================
# تاب 2: روابط المناديب
# ==================================================================================
with tab2:
    st.info("⚠️ الروابط دي بتتغير كل يوم تلقائياً — ابعتها للمناديب كل صباح على واتساب.")

    # زر لإنشاء رسالة واتساب جماعية
    base_url = st.text_input(
        "🌐 رابط الموقع الأساسي",
        value="https://your-app.streamlit.app",
        help="الرابط بتاع تطبيقك على Streamlit Cloud"
    )

    st.markdown("### روابط اليوم لكل المناديب")

    for driver in DRIVERS:
        token = get_driver_token(driver, today)
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
    wa_msg = f"🚚 *تقرير الطلبات - {datetime.now().strftime('%d/%m/%Y')}*\n\n"
    wa_msg += "كل مندوب يفتح رابطه ويحدث عدد طلباته:\n\n"
    for driver in DRIVERS:
        token = get_driver_token(driver, today)
        wa_msg += f"▫️ {driver}: {base_url}?driver={token}\n"
    st.text_area("رسالة واتساب", wa_msg, height=300)

# ==================================================================================
# تاب 3: الأرشيف
# ==================================================================================
with tab3:
    st.subheader("📁 سجل الأيام السابقة")

    available_dates = sorted(all_data.keys(), reverse=True)
    if not available_dates:
        st.info("مفيش بيانات محفوظة لحد دلوقتي.")
    else:
        selected_date = st.selectbox("اختار يوم", available_dates)
        day_data = all_data.get(selected_date, {})

        rows = []
        for driver in DRIVERS:
            entry = day_data.get(driver, {})
            rows.append({
                "اسم المندوب": driver,
                "عدد الطلبات": entry.get("count", 0),
                "وقت التحديث": entry.get("time", "-")
            })
        df_archive = pd.DataFrame(rows)
        st.dataframe(df_archive, use_container_width=True, hide_index=True)

        # تصدير الأرشيف
        output2 = BytesIO()
        with pd.ExcelWriter(output2, engine="xlsxwriter") as writer:
            df_archive.to_excel(writer, index=False, sheet_name=selected_date)
        st.download_button(
            label=f"⬇️ تحميل تقرير {selected_date}",
            data=output2.getvalue(),
            file_name=f"مناديب_{selected_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
