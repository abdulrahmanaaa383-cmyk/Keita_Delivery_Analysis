import streamlit as st
import pandas as pd
import json
import os
from datetime import date, datetime
from io import BytesIO

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

# ==================================================================================
# تحميل البيانات
# ==================================================================================
drivers_list = load_json(DRIVERS_FILE, [
    "أحمد محمد", "محمود علي", "عمر حسن", "يوسف إبراهيم", "مصطفى عبدالله"
])
all_data = load_json(DATA_FILE, {})
today = get_today()
if today not in all_data:
    all_data[today] = {}

ADMIN_PASSWORD = "admin123"  # <-- غيّر الباسورد هنا

# ==================================================================================
# CSS
# ==================================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
* { font-family: 'Cairo', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; }
.stNumberInput > div > div > input {
    text-align: center;
    font-size: 1.2rem;
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
# تحديد الوضع
# ==================================================================================
mode = st.query_params.get("mode", "driver")

# ==================================================================================
# ██████  صفحة المناديب  ██████
# ==================================================================================
if mode != "admin":

    st.markdown("## 🚚 إدخال الطلبات اليومية")
    st.caption(f"📅 {datetime.now().strftime('%A %d/%m/%Y')}")
    st.info("كل مندوب يكتب عدد طلباته جنب اسمه ويضغط **حفظ الكل** في الآخر")

    st.markdown("---")

    # ---- رأس الجدول ----
    h1, h2, h3 = st.columns([3, 2, 2])
    h1.markdown("**اسم المندوب**")
    h2.markdown("**عدد الطلبات**")
    h3.markdown("**آخر تحديث**")

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
        changed = False
        for driver, count in inputs.items():
            old = all_data[today].get(driver, {}).get("count", 0)
            if count != old or count > 0:
                all_data[today][driver] = {"count": count, "time": now_time}
                changed = True
        if changed:
            save_json(DATA_FILE, all_data)
        st.success("✅ تم الحفظ! البيانات ظهرت عند المدير دلوقتي.")
        st.rerun()

    st.markdown("---")
    st.caption("🔐 [دخول المدير](?mode=admin)")


# ==================================================================================
# ██████  صفحة المدير  ██████
# ==================================================================================
else:
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
    st.markdown("## 🗂️ داشبورد المدير")
    st.caption(f"📅 {datetime.now().strftime('%A %d/%m/%Y %H:%M')}")

    today_data = all_data.get(today, {})
    total  = sum(v.get("count", 0) for v in today_data.values())
    active = sum(1 for v in today_data.values() if v.get("count", 0) > 0)
    avg    = round(total / active, 1) if active > 0 else 0
    top    = max(today_data, key=lambda d: today_data[d].get("count", 0), default="-")
    top_v  = today_data.get(top, {}).get("count", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 إجمالي الطلبات", total)
    c2.metric("👤 سجّلوا اليوم",    f"{active} / {len(drivers_list)}")
    c3.metric("📊 متوسط / مندوب",   avg)
    c4.metric("🏆 الأعلى أداء",     f"{top.split()[0] if top != '-' else '-'} ({top_v})")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 طلبات اليوم", "✏️ إدارة المناديب", "📁 الأرشيف"])

    # ----------------------------------------------------------
    # تاب 1: طلبات اليوم
    # ----------------------------------------------------------
    with tab1:
        _, col_ref = st.columns([4, 1])
        with col_ref:
            if st.button("🔄 تحديث", use_container_width=True):
                st.rerun()

        rows = []
        for driver in drivers_list:
            e = today_data.get(driver, {})
            rows.append({
                "اسم المندوب": driver,
                "عدد الطلبات": e.get("count", 0),
                "آخر تحديث":   e.get("time", "—"),
                "الحالة":      "✅ سجّل" if e.get("count", 0) > 0 else "⏳ لم يسجّل"
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={"عدد الطلبات": st.column_config.NumberColumn(format="%d 📦")}
        )

        # تصدير Excel
        output = BytesIO()
        export_df = df.copy()
        total_row = pd.DataFrame([{
            "اسم المندوب": "الإجمالي",
            "عدد الطلبات": df["عدد الطلبات"].sum(),
            "آخر تحديث": "",
            "الحالة": ""
        }])
        export_df = pd.concat([export_df, total_row], ignore_index=True)
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, index=False, sheet_name=today)
            wb = writer.book
            ws = writer.sheets[today]
            hfmt = wb.add_format({
                'bold': True, 'bg_color': '#1D9E75',
                'font_color': 'white', 'border': 1, 'align': 'center'
            })
            for i, col in enumerate(export_df.columns):
                ws.write(0, i, col, hfmt)
            ws.set_column(0, 0, 25)
            ws.set_column(1, 3, 18)

        st.download_button(
            "⬇️ تحميل Excel اليوم",
            data=output.getvalue(),
            file_name=f"مناديب_{today}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        if st.button("🗑️ تصفير طلبات اليوم", type="secondary"):
            all_data[today] = {}
            save_json(DATA_FILE, all_data)
            st.success("تم التصفير!")
            st.rerun()

    # ----------------------------------------------------------
    # تاب 2: إدارة المناديب
    # ----------------------------------------------------------
    with tab2:
        st.markdown("### إضافة مناديب جدد")
        new_names_input = st.text_area(
            "اكتب كل اسم في سطر منفصل",
            placeholder="محمد إبراهيم\nعلي حسن\nكريم سامي",
            height=150
        )
        if st.button("➕ إضافة", type="primary"):
            new_names = [n.strip() for n in new_names_input.splitlines() if n.strip()]
            added = 0
            for name in new_names:
                if name not in drivers_list:
                    drivers_list.append(name)
                    added += 1
            save_json(DRIVERS_FILE, drivers_list)
            st.success(f"تم إضافة {added} مندوب!")
            st.rerun()

        st.markdown("---")
        st.markdown("### تعديل أو حذف مندوب")

        to_delete = []
        for i, driver in enumerate(drivers_list):
            col_n, col_e, col_d = st.columns([1, 4, 1])
            with col_n:
                st.markdown(
                    f"<div style='padding:10px 0; color:#6c757d; font-size:0.85rem;'>{i+1}</div>",
                    unsafe_allow_html=True
                )
            with col_e:
                new_name = st.text_input(
                    f"اسم {i+1}",
                    value=driver,
                    key=f"edit_{i}",
                    label_visibility="collapsed"
                )
                drivers_list[i] = new_name
            with col_d:
                if st.button("🗑️", key=f"del_{i}"):
                    to_delete.append(driver)

        if to_delete:
            for d in to_delete:
                if d in drivers_list:
                    drivers_list.remove(d)
            save_json(DRIVERS_FILE, drivers_list)
            st.rerun()

        if st.button("💾 حفظ التعديلات", type="secondary", use_container_width=True):
            save_json(DRIVERS_FILE, drivers_list)
            st.success("تم الحفظ!")

    # ----------------------------------------------------------
    # تاب 3: الأرشيف
    # ----------------------------------------------------------
    with tab3:
        dates = sorted(all_data.keys(), reverse=True)
        if not dates:
            st.info("مفيش بيانات محفوظة لحد دلوقتي.")
        else:
            selected = st.selectbox("اختار يوم", dates)
            day_data = all_data.get(selected, {})
            arch_rows = []
            for driver in drivers_list:
                e = day_data.get(driver, {})
                arch_rows.append({
                    "اسم المندوب": driver,
                    "عدد الطلبات": e.get("count", 0),
                    "وقت التحديث": e.get("time", "—")
                })
            df_arch = pd.DataFrame(arch_rows)
            st.dataframe(df_arch, use_container_width=True, hide_index=True)

            out2 = BytesIO()
            with pd.ExcelWriter(out2, engine="xlsxwriter") as writer:
                df_arch.to_excel(writer, index=False, sheet_name=selected)
            st.download_button(
                f"⬇️ تحميل {selected}",
                data=out2.getvalue(),
                file_name=f"مناديب_{selected}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.markdown("---")
    if st.button("🚪 خروج"):
        st.session_state["admin_logged_in"] = False
        st.rerun()
