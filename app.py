import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np

# ==============================================================================
# 1. الدوال المساعدة لتحميل ومعالجة البيانات
# ==============================================================================

def clean_and_process_data(df):
    """
    تنظيف وتوحيد أسماء الأعمدة وتحويل البيانات للتحليل.
    """
    
    # تنظيف أسماء الأعمدة من المسافات الزائدة
    df.columns = df.columns.str.strip()
    
    # التحديد الدقيق لخرائط الأسماء من ملف المستخدم إلى الأسماء الداخلية للكود 
    COLUMN_MAPPING = {
        'Courier ID': 'ID',
        'Courier First Name': 'First Name',
        'Courier Last Name': 'Last Name',
        'Valid Online Time': 'Online Time (h)',
        'Delivered Tasks': 'Delivered Tasks',
        'Cancelled Tasks': 'Cancelled Tasks',
        'Rejected Tasks': 'Rejected Tasks',
        'On-time Rate (D)': 'On-time Rate',
        'Avg Delivery Time of Delivered Orders': 'Avg Delivery Time (min)',
        'Cancellation Rate from Delivery Issues': 'Cancellation Rate'
    }
    
    # إعادة تسمية الأعمدة الموجودة في الملف إلى الأسماء القياسية التي يستخدمها الكود
    df = df.rename(columns=COLUMN_MAPPING, errors='ignore')

    # التأكد من تحويل الأعمدة الرقمية إلى النوع float باستخدام الأسماء الداخلية الجديدة
    for col in [
        'Online Time (h)', 'Delivered Tasks', 'On-time Rate', 
        'Avg Delivery Time (min)', 'Cancellation Rate',
        'Cancelled Tasks', 'Rejected Tasks'
    ]:
        if col in df.columns:
            # معالجة القيم التي قد تكون نسب مئوية أو سلاسل نصية
            df[col] = df[col].astype(str).str.replace('[^0-9.+-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # تصفية الصفوف التي لا تحتوي على ID للمندوب أو ساعات عمل فعالة
    df = df.dropna(subset=['ID'])
    if 'Online Time (h)' in df.columns:
         df = df[df['Online Time (h)'] > 0].reset_index(drop=True)
    
    return df

def generate_pivot_table(df):
    """ينشئ الجدول المحوري (Pivot Table) بتجميع مؤشرات الأداء."""
    
    # تجميع البيانات حسب المندوب
    pivot_df = df.groupby(['ID', 'First Name', 'Last Name']).agg(
        Total_Delivered_Tasks=('Delivered Tasks', 'sum'),
        Total_Online_Hours=('Online Time (h)', 'sum'),
        Total_Cancelled_Tasks=('Cancelled Tasks', 'sum'),
        Total_Rejected_Tasks=('Rejected Tasks', 'sum'),
        Avg_On_time_Rate=('On-time Rate', 'mean'),
        Avg_Delivery_Time=('Avg Delivery Time (min)', 'mean'),
        Avg_Cancellation_Rate=('Cancellation Rate', 'mean')
    ).reset_index()

    # إنشاء عمود الاسم الكامل
    pivot_df['Agent Name'] = pivot_df['First Name'] + ' ' + pivot_df['Last Name']

    # حساب مؤشر الإنتاجية الأساسي: عدد الطلبات في الساعة (Tasks Per Hour)
    # التأكد من عدم القسمة على صفر
    pivot_df['Tasks Per Hour'] = np.where(
        pivot_df['Total_Online_Hours'] > 0,
        (pivot_df['Total_Delivered_Tasks'] / pivot_df['Total_Online_Hours']),
        0
    ).round(2)
    
    # تنسيق النسب المئوية 
    pivot_df['Avg_On_time_Rate (%)'] = (pivot_df['Avg_On_time_Rate'] * 100).round(2).astype(str) + '%'
    pivot_df['Avg_Cancellation_Rate (%)'] = (pivot_df['Avg_Cancellation_Rate'] * 100).round(2).astype(str) + '%'
    
    # إعادة ترتيب الأعمدة للعرض النهائي (مع إضافة الملغاة والمرفوضة)
    pivot_df = pivot_df[['ID', 'Agent Name', 
                         'Total_Delivered_Tasks', 'Total_Online_Hours', 'Tasks Per Hour',
                         'Total_Cancelled_Tasks', 'Total_Rejected_Tasks',
                         'Avg_On_time_Rate (%)', 'Avg_Delivery_Time', 'Avg_Cancellation_Rate (%)']]
    
    # 🌟 تحديث: إعادة تسمية الأعمدة للعرض باللغة الإنجليزية 🌟
    display_cols = {
        'Total_Delivered_Tasks': 'Total Delivered Tasks',
        'Total_Online_Hours': 'Total Online Hours (h)',
        'Tasks Per Hour': 'Tasks Per Hour (TPH)',
        'Total_Cancelled_Tasks': 'Total Cancelled Tasks',
        'Total_Rejected_Tasks': 'Total Rejected Tasks',
        'Avg_On_time_Rate (%)': 'Avg On-time Rate (%)',
        'Avg_Delivery_Time': 'Avg Delivery Time (min)',
        'Avg_Cancellation_Rate (%)': 'Avg Cancellation Rate (%)'
    }
    
    # المتغير الذي يحمل الجدول المنسق للعرض في Streamlit
    display_df = pivot_df.rename(columns=display_cols).drop(columns=['First Name', 'Last Name'], errors='ignore')
    
    return pivot_df, display_df

def analyze_performance(pivot_df):
    """تطبيق منطق العمل لإنشاء توصيات بناءً على المقارنة بالمتوسط."""
    recommendations = {}

    analysis_df = pivot_df.copy()
    
    # تحويل النسب المئوية إلى أرقام للتحليل
    analysis_df['On_time_Rate_Num'] = analysis_df['Avg_On_time_Rate (%)'].str.replace('%', '').astype(float) / 100
    analysis_df['Cancellation_Rate_Num'] = analysis_df['Avg_Cancellation_Rate (%)'].str.replace('%', '').astype(float) / 100
    
    # حساب المتوسطات للمقارنة
    avg_ontime = analysis_df['On_time_Rate_Num'].mean()
    avg_delivery_time = analysis_df['Avg_Delivery_Time'].mean()
    avg_cancellation_rate = analysis_df['Cancellation_Rate_Num'].mean()
    avg_tph = analysis_df['Tasks Per Hour'].mean()
    avg_cancelled_count = analysis_df['Total_Cancelled_Tasks'].mean()
    avg_rejected_count = analysis_df['Total_Rejected_Tasks'].mean()
    
    # تعريف الحدود الدنيا/القصوى
    LOW_PERFORMANCE_THRESHOLD = 0.8 
    HIGH_PERFORMANCE_THRESHOLD = 1.2 

    for index, row in analysis_df.iterrows():
        agent_name = row['Agent Name']
        notes = []

        # 1. تحليل كفاءة التسليم والالتزام بالوقت
        if row['On_time_Rate_Num'] < (avg_ontime * LOW_PERFORMANCE_THRESHOLD):
            notes.append(f"🔴 Low On-Time Rate: {row['Avg_On_time_Rate (%)']} — Needs path management improvement.")
        
        # 2. تحليل سرعة التسليم
        if row['Avg_Delivery_Time'] > (avg_delivery_time * HIGH_PERFORMANCE_THRESHOLD) and row['Total_Delivered_Tasks'] > 0:
            notes.append(f"🟡 High Delivery Time: {row['Avg_Delivery_Time']:.2f} min — Needs speed/movement improvement.")
        
        # 3. تحليل معدل الإلغاء (النسبة)
        if row['Cancellation_Rate_Num'] > (avg_cancellation_rate * HIGH_PERFORMANCE_THRESHOLD) and row['Cancellation_Rate_Num'] * 100 > 2:
            notes.append(f"❌ High Cancellation Rate (Ratio): {row['Avg_Cancellation_Rate (%)']} — Requires review of cancellation reasons.")
        
        # 4. تحليل إجمالي الطلبات الملغاة (العدد)
        if row['Total_Cancelled_Tasks'] > (avg_cancelled_count * HIGH_PERFORMANCE_THRESHOLD) and row['Total_Cancelled_Tasks'] >= 5:
             notes.append(f"🔥 High Total Cancellations: {int(row['Total_Cancelled_Tasks'])} tasks. Review task acceptance behavior or location/communication issues.")

        # 5. تحليل إجمالي الطلبات المرفوضة (العدد)
        if row['Total_Rejected_Tasks'] > (avg_rejected_count * HIGH_PERFORMANCE_THRESHOLD) and row['Total_Rejected_Tasks'] >= 10:
             notes.append(f"🛑 High Total Rejections: {int(row['Total_Rejected_Tasks'])} tasks. May indicate hesitation in accepting tasks or negative perception of certain areas.")
        
        # 6. الإنتاجية
        if row['Tasks Per Hour'] < (avg_tph * LOW_PERFORMANCE_THRESHOLD) and row['Total_Online_Hours'] > 5:
            notes.append(f"📉 Low Productivity: {row['Tasks Per Hour']:.2f} TPH — Recommend working during peak hours or reviewing waiting process.")

        if notes:
            recommendations[agent_name] = {'ID': row['ID'], 'Notes': notes}

    return recommendations

def to_excel(df):
    """دالة لتحويل DataFrame إلى ملف Excel في الذاكرة."""
    output = BytesIO()
    
    # إعادة تسمية الأعمدة النهائية في ملف الإكسيل بالإنجليزية
    export_df = df.copy()
    # 🌟 تحديث: أسماء الأعمدة للتصدير بالإنجليزية 🌟
    english_cols = {
        'ID': 'Courier ID',
        'Agent Name': 'Agent Full Name',
        'Total_Delivered_Tasks': 'Total Delivered Tasks',
        'Total_Online_Hours': 'Total Online Hours (h)',
        'Tasks Per Hour': 'Tasks Per Hour (TPH)',
        'Total_Cancelled_Tasks': 'Total Cancelled Tasks',
        'Total_Rejected_Tasks': 'Total Rejected Tasks',
        'Avg_On_time_Rate (%)': 'Avg On-time Rate (%)',
        'Avg_Delivery_Time': 'Avg Delivery Time (min)',
        'Avg_Cancellation_Rate (%)': 'Avg Cancellation Rate (%)'
    }
    
    export_df = export_df.rename(columns=english_cols)
    
    # تنسيق الأعمدة الرقمية قبل التصدير
    for col in ['Total Delivered Tasks', 'Total Cancelled Tasks', 'Total Rejected Tasks']:
        if col in export_df.columns:
            export_df[col] = export_df[col].round(0).astype(int)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Keita_Performance_Report')
        
    return output.getvalue()

# ==============================================================================
# 2. واجهة التطبيق (Streamlit)
# ==============================================================================

st.set_page_config(layout="wide", page_title="Advanced Delivery Agent Performance Analyzer (Keita)")
st.title("🛵 Advanced Delivery Agent Performance Analyzer (Keita)")
st.markdown("---")

uploaded_file = st.file_uploader("📥 Please upload your Excel/CSV file for performance analysis", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # قراءة الملف (مع افتراض أن الملف قد يحتوي على CSV أو Excel)
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"File loaded successfully: {uploaded_file.name} — Records: {len(df)}")
        
        # 1. تنظيف وإعادة تسمية الأعمدة
        df = clean_and_process_data(df)
        
        if df.empty:
            st.error("❌ No valid data for analysis (Ensure agents have active online hours).")
            st.stop()
            
        st.subheader("📋 Processed Data (First 5 Rows)")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        st.markdown("---")

        # 2. إنشاء الجدول المحوري
        pivot_df, display_pivot = generate_pivot_table(df)
        
        st.header("📈 Consolidated Performance Report")
        
        # عرض الجدول المحوري المنسق بالإنجليزية
        st.dataframe(display_pivot.style.format({
            'Total Online Hours (h)': '{:.2f}',
            'Tasks Per Hour (TPH)': '{:.2f}',
            'Avg Delivery Time (min)': '{:.2f}'
        }), use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Export Detailed Excel Report",
            data=to_excel(pivot_df),
            file_name="Keita_Delivery_Report_EN.xlsx",
            mime="application/vnd.ms-excel"
        )

        st.markdown("---")

        st.header("📝 Recommendations and Behavioral Analysis")
        recommendations = analyze_performance(pivot_df)

        if recommendations:
            st.warning(f"⚠️ **Alert:** **{len(recommendations)}** agents identified with below-average performance or behavioral issues (High Cancellation/Rejection):")
            for agent, data in recommendations.items():
                st.markdown(f"### 👤 Agent: {agent} (ID: {data['ID']})")
                for note in data['Notes']:
                    st.markdown(f"- {note}")
                st.markdown("---")
        else:
            st.balloons()
            st.success("🎉 **No immediate issues found!** Overall performance is within acceptable limits.")

    except Exception as e:
        # عرض رسالة خطأ أكثر فائدة في حال وجود أي خطأ آخر
        st.error(f"❌ An error occurred while reading or processing the file. Please ensure column names and numeric data format are correct.")
        # هنا نعرض الخطأ الفني في الواجهة لمساعدتنا في التصحيح
        st.exception(e)
else:
    st.info("Upload the file to start the performance analysis.")
