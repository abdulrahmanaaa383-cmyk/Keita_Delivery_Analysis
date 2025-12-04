import streamlit as st
import pandas as pd
from io import BytesIO

# ==============================================================================
# 1. الدوال المساعدة لتحميل ومعالجة البيانات
# ==============================================================================

def clean_and_process_data(df):
    """
    تنظيف وتوحيد أسماء الأعمدة وتحويل البيانات للتحليل.
    """
    
    # تنظيف أسماء الأعمدة من المسافات الزائدة
    df.columns = df.columns.str.strip()
    
    # تحديد الأعمدة الأساسية المطلوبة لتحليل مناديب كيتا
    required_cols = {
        'Courier ID': 'ID',
        'Courier First Name': 'First Name',
        'Courier Last Name': 'Last Name',
        'Valid Online Time': 'Online Time (h)',
        'Delivered Tasks': 'Delivered Tasks',
        'On-time Rate (D)': 'On-time Rate',
        'Avg Delivery Time of Delivered Orders': 'Avg Delivery Time (min)',
        'Cancellation Rate from Delivery Issues': 'Cancellation Rate'
    }
    
    # إعادة تسمية الأعمدة الموجودة في الملف إلى الأسماء القياسية
    current_cols = {c: required_cols[c] for c in required_cols if c in df.columns}
    # التأكد من وجود الأعمدة الأساسية الثلاثة على الأقل قبل المتابعة
    if not all(col in df.columns for col in ['ID', 'First Name', 'Last Name']):
        # إذا لم يكن هناك تطابق، نستخدم التسميات الافتراضية إذا كانت موجودة
        df = df.rename(columns=current_cols, errors='ignore')
    else:
         df = df.rename(columns=current_cols)


    # التأكد من تحويل الأعمدة الرقمية إلى النوع float
    for col in ['Online Time (h)', 'Delivered Tasks', 'On-time Rate', 'Avg Delivery Time (min)', 'Cancellation Rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # تصفية الصفوف التي لا تحتوي على ID للمندوب
    df = df.dropna(subset=['ID'])
    
    return df

def generate_pivot_table(df):
    """ينشئ الجدول المحوري (Pivot Table) بتجميع مؤشرات الأداء."""
    
    # تجميع البيانات حسب المندوب
    pivot_df = df.groupby(['ID', 'First Name', 'Last Name']).agg(
        Total_Delivered_Tasks=('Delivered Tasks', 'sum'),
        Total_Online_Hours=('Online Time (h)', 'sum'),
        Avg_On_time_Rate=('On-time Rate', 'mean'),
        Avg_Delivery_Time=('Avg Delivery Time (min)', 'mean'),
        Avg_Cancellation_Rate=('Cancellation Rate', 'mean')
    ).reset_index()

    # إنشاء عمود الاسم الكامل
    pivot_df['Agent Name'] = pivot_df['First Name'] + ' ' + pivot_df['Last Name']

    # حساب مؤشر الإنتاجية الأساسي: عدد الطلبات في الساعة (Tasks Per Hour)
    pivot_df['Tasks Per Hour'] = (pivot_df['Total_Delivered_Tasks'] / pivot_df['Total_Online_Hours']).fillna(0).round(2)
    
    # تنسيق النسب المئوية 
    pivot_df['Avg_On_time_Rate (%)'] = (pivot_df['Avg_On_time_Rate'] * 100).round(2).astype(str) + '%'
    pivot_df['Avg_Cancellation_Rate (%)'] = (pivot_df['Avg_Cancellation_Rate'] * 100).round(2).astype(str) + '%'
    
    # إعادة ترتيب الأعمدة للعرض النهائي
    pivot_df = pivot_df[['ID', 'Agent Name', 'Total_Delivered_Tasks', 'Total_Online_Hours', 'Tasks Per Hour',
                         'Avg_On_time_Rate (%)', 'Avg_Delivery_Time', 'Avg_Cancellation_Rate (%)']]
    
    return pivot_df

def analyze_performance(pivot_df):
    """تطبيق منطق العمل لإنشاء توصيات بناءً على المقارنة بالمتوسط."""
    recommendations = {}

    analysis_df = pivot_df.copy()
    analysis_df['On_time_Rate_Num'] = analysis_df['Avg_On_time_Rate (%)'].str.replace('%', '').astype(float) / 100
    analysis_df['Cancellation_Rate_Num'] = analysis_df['Avg_Cancellation_Rate (%)'].str.replace('%', '').astype(float) / 100
    
    # حساب المتوسطات للمقارنة
    avg_ontime = analysis_df['On_time_Rate_Num'].mean()
    avg_delivery_time = analysis_df['Avg_Delivery_Time'].mean()
    avg_cancellation = analysis_df['Cancellation_Rate_Num'].mean()
    avg_tph = analysis_df['Tasks Per Hour'].mean()
    
    # تعريف الحدود الدنيا/القصوى
    LOW_PERFORMANCE_THRESHOLD = 0.8 
    HIGH_PERFORMANCE_THRESHOLD = 1.2 

    for index, row in analysis_df.iterrows():
        agent_name = row['Agent Name']
        notes = []

        # 1. تحليل كفاءة التسليم والالتزام بالوقت
        if row['On_time_Rate_Num'] < (avg_ontime * LOW_PERFORMANCE_THRESHOLD):
            notes.append(f"🔴 انخفاض الالتزام بالوقت: {row['Avg_On_time_Rate (%)']} — يحتاج لتحسين إدارة المسارات.")
        
        # 2. تحليل سرعة التسليم
        if row['Avg_Delivery_Time'] > (avg_delivery_time * HIGH_PERFORMANCE_THRESHOLD) and row['Total_Delivered_Tasks'] > 0:
            notes.append(f"🟡 وقت التسليم مرتفع: {row['Avg_Delivery_Time']:.2f} دقيقة — يحتاج لتحسين سرعة الحركة.")
        
        # 3. تحليل معدل الإلغاء
        if row['Cancellation_Rate_Num'] > (avg_cancellation * HIGH_PERFORMANCE_THRESHOLD) and row['Cancellation_Rate_Num'] > 0.05:
            notes.append(f"❌ معدل إلغاء مرتفع: {row['Avg_Cancellation_Rate (%)']} — يتطلب مراجعة أسباب الإلغاء.")
        
        # 4. الإنتاجية
        if row['Tasks Per Hour'] < (avg_tph * LOW_PERFORMANCE_THRESHOLD) and row['Total_Online_Hours'] > 5:
            notes.append(f"📉 إنتاجية منخفضة: {row['Tasks Per Hour']:.2f} طلب/ساعة — يفضل العمل في ساعات الذروة.")

        if notes:
            recommendations[agent_name] = {'ID': row['ID'], 'Notes': notes}

    return recommendations

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Keeta_Delivery_Report', float_format='%.2f')
    return output.getvalue()

# ==============================================================================
# 2. واجهة التطبيق (Streamlit)
# ==============================================================================

st.set_page_config(layout="wide", page_title="أداة تحليل أداء مناديب كيتا")
st.title("🛵 محلل أداء مناديب التوصيل المتقدم (كيتا)")
st.markdown("---")

uploaded_file = st.file_uploader("📥 يرجى رفع ملف الإكسيل/CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"تم تحميل الملف: {uploaded_file.name} — السجلات: {len(df)}")
        
        df = clean_and_process_data(df)
        
        st.subheader("📋 البيانات بعد المعالجة (أول 5 صفوف)")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("---")

        pivot_table = generate_pivot_table(df)
        
        display_pivot = pivot_table.drop(columns=['First Name', 'Last Name'], errors='ignore')
        st.dataframe(display_pivot, use_container_width=True)

        st.download_button(
            label="⬇️ تصدير Excel",
            data=to_excel(pivot_table),
            file_name="Keeta_Delivery_Report.xlsx",
            mime="application/vnd.ms-excel"
        )

        st.markdown("---")

        st.header("📝 التوصيات")
        recommendations = analyze_performance(pivot_table)

        if recommendations:
            st.warning("⚠️ تم تحديد مناديب بحاجة لتحسين:")
            for agent, data in recommendations.items():
                st.markdown(f"### المندوب: {agent} (ID: {data['ID']})")
                for note in data['Notes']:
                    st.markdown(f"- {note}")
                st.markdown("---")
        else:
            st.balloons()
            st.success("🎉 لا توجد مشاكل واضحة!")

    except Exception as e:
        st.error(f"❌ خطأ أثناء قراءة الملف: {e}")
else:
    st.info("قم برفع الملف للبدء.")
