import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np

# ==============================================================================
# 1. تحديد الثوابت وتحديد حساسية الأداء
# ==============================================================================

# *** التحكم في حساسية التلوين والتوصيات ***
# القيمة التالية تحدد متى نعتبر الأداء سيئًا مقارنة بالمتوسط.
# 0.90 تعني أن الأداء سيئ إذا كان أقل من 90% من المتوسط (أي أقل بـ 10%)
PERFORMANCE_THRESHOLD = 0.90 

# قائمة الأعمدة الأصلية المطلوبة للتحليل في التقرير النهائي
# ملاحظة: سنستخدم هذه الأسماء في التقرير بدلاً من الأسماء المعربة
# سنقوم بتغيير هذه القائمة لتكون مفتاحاً مرناً للبحث عن الأعمدة
REQUIRED_COLS_MAPPING = {
    # الأعمدة التعريفية التي يجب أن تكون موجودة كما هي
    'Courier ID': 'Courier ID',
    'Courier First Name': 'First Name',
    'Courier Last Name': 'Last Name',
    
    # مؤشرات الأداء المطلوبة للتحليل (سنستخدم جزء من الاسم الأصلي للبحث المرن)
    'Valid Online Time': 'Valid Online Time', # ساعات العمل الفعالة
    'Courier App Online Time': 'Courier App Online Time', # وقت الاتصال بالتطبيق
    'Accepted Tasks': 'Accepted Tasks',
    'Delivered Tasks': 'Delivered Tasks',
    'Cancelled Tasks': 'Cancelled Tasks',
    'Rejected Tasks': 'Rejected Tasks',
    'On-time Rate (D)': 'On-time Rate (D)', # معدل الالتزام
    'Avg Delivery Time of Delivered Orders': 'Avg Delivery Time of Delivered Orders', # متوسط وقت التسليم
    'Cancellation Rate from Delivery Issues': 'Cancellation Rate from Delivery Issues' # معدل الإلغاء
}

# ==============================================================================
# 2. الدوال المساعدة لتحميل ومعالجة البيانات
# ==============================================================================

def clean_and_process_data(df):
    """
    تنظيف وتوحيد أسماء الأعمدة وتحويل البيانات للتحليل.
    ** تم التعديل ليكون مرناً في تحديد الأعمدة **
    """
    
    # 1. تنظيف وتوحيد أسماء الأعمدة المتاحة في الملف
    df.columns = df.columns.astype(str).str.strip()
    original_cols = df.columns.tolist()
    
    # 2. إنشاء خريطة البحث المرن (Normalized Map)
    # نستخدم جزء من الاسم ليكون مرجعاً للبحث (مثلاً 'online time' يجب أن يجد 'Courier App Online Time')
    # يجب أن تكون المفاتيح بالصيغة الموحدة (lowercase, no spaces)
    normalized_cols_map = {col.lower().replace(' ', ''): col for col in original_cols}
    
    # 3. تحديد الأعمدة المطلوبة فعلياً وإعادة تسميتها
    found_cols = {}
    missing_cols_names = []
    
    # مفاتيح البحث المرنة
    search_keys = {
        'courierid': 'Courier ID',
        'courierfirstname': 'Courier First Name',
        'courierlastname': 'Courier Last Name',
        'validonlinetime': 'Valid Online Time',
        'courierapponlinetime': 'Courier App Online Time',
        'acceptedtasks': 'Accepted Tasks',
        'deliveredtasks': 'Delivered Tasks',
        'cancelledtasks': 'Cancelled Tasks',
        'rejectedtasks': 'Rejected Tasks',
        'ontimerated': 'On-time Rate (D)', # البحث عن 'On-time Rate (D)'
        'avgdeliverytimeofdeliveredorders': 'Avg Delivery Time of Delivered Orders',
        'cancellationratefromdeliveryissues': 'Cancellation Rate from Delivery Issues'
    }
    
    for search_key, required_name in search_keys.items():
        # نبحث عن الاسم في الخريطة الموحدة
        found = False
        for normalized_col_name, original_col_name in normalized_cols_map.items():
            if search_key in normalized_col_name:
                found_cols[original_col_name] = required_name
                found = True
                break
        
        # إذا لم يتم العثور عليه، نضيفه لقائمة المفقودات إذا كان أساسياً (باستثناء الاسم الأول والأخير حيث يمكن أن يكونا غير موجودين)
        if not found and required_name in REQUIRED_COLS_MAPPING:
             # إذا كان عموداً حاسماً للحسابات (مثل الوقت والإنتاجية)، نعتبره مفقوداً
            if required_name not in ['Courier First Name', 'Courier Last Name']:
                 missing_cols_names.append(required_name)

    # 4. رفع خطأ إذا كانت الأعمدة الحاسمة مفقودة
    if missing_cols_names:
        raise ValueError(f"الملف لا يحتوي على الأعمدة الأساسية اللازمة للتحليل: {', '.join(missing_cols_names)}. يرجى التحقق من رؤوس الأعمدة.")
    
    # 5. تصفية وإعادة تسمية الأعمدة
    df = df[found_cols.keys()].rename(columns=found_cols)
    
    # 6. تحويل البيانات إلى أرقام (باستخدام الأسماء الجديدة)
    numeric_cols = [
        'Courier App Online Time', 'Valid Online Time', 'Accepted Tasks', 
        'Delivered Tasks', 'Cancelled Tasks', 'Rejected Tasks', 
        'On-time Rate (D)', 'Avg Delivery Time of Delivered Orders', 
        'Cancellation Rate from Delivery Issues'
    ]

    for col in numeric_cols:
        if col in df.columns:
            # تحويل القيم التي قد تكون في شكل سلاسل نصية (مثل 30.5h) إلى أرقام
            df[col] = df[col].astype(str).str.replace('[^0-9.+-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 7. تصفية الصفوف التي لا تحتوي على ID للمندوب
    if 'Courier ID' in df.columns:
        df = df.dropna(subset=['Courier ID'])
    
    # 8. 🔴 فلترة المناديب الذين لم يعملوا (ساعات الأونلاين الفعالة 0)
    if 'Valid Online Time' in df.columns:
        df = df[df['Valid Online Time'] > 0].reset_index(drop=True)
    
    # التأكد من وجود أعمدة الاسم (إذا كانت مفقودة نضع قيمة فارغة)
    if 'Courier First Name' not in df.columns:
        df['Courier First Name'] = ''
    if 'Courier Last Name' not in df.columns:
        df['Courier Last Name'] = ''

    return df

def generate_pivot_table(df):
    """ينشئ الجدول المحوري (Pivot Table) بتجميع مؤشرات الأداء المطلوبة."""
    
    # الأعمدة الأساسية للتجميع (مفاتيح الصف)
    group_cols = ['Courier ID', 'Courier First Name', 'Courier Last Name']
    
    # الأعمدة المتاحة للحساب
    available_cols = [col for col in df.columns if col not in group_cols]
    
    # قاموس التجميع (Aggregation Dictionary) بناءً على الأعمدة المتاحة
    agg_dict = {}
    
    # تحديد وظيفة التجميع لكل عمود
    for col in available_cols:
        if 'Time' in col or 'Tasks' in col:
            agg_dict[col] = 'sum'
        elif 'Rate' in col or 'Avg' in col:
            agg_dict[col] = 'mean'
            
    # التحقق من وجود الأعمدة الأساسية قبل التجميع
    if not agg_dict:
        # إذا لم تكن هناك أعمدة للقياس، نرجع DataFrame فارغاً
        return pd.DataFrame()
        
    pivot_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # إنشاء عمود الاسم الكامل
    pivot_df['Agent Name'] = pivot_df['Courier First Name'].fillna('') + ' ' + pivot_df['Courier Last Name'].fillna('')
    
    # 🌟 إضافة مؤشر TPH (الإنتاجية) كأهم مؤشر جديد
    if 'Delivered Tasks' in pivot_df.columns and 'Valid Online Time' in pivot_df.columns:
        pivot_df['TPH (Tasks Per Valid Hour)'] = np.where(
            pivot_df['Valid Online Time'] > 0,
            (pivot_df['Delivered Tasks'] / pivot_df['Valid Online Time']),
            0
        ).round(2)
    else:
        pivot_df['TPH (Tasks Per Valid Hour)'] = 0

    
    # ترتيب الأعمدة للعرض النهائي
    final_cols = [
        'Courier ID', 'Agent Name', 
        'Valid Online Time', 'Courier App Online Time',
        'TPH (Tasks Per Valid Hour)',
        'Delivered Tasks', 'Accepted Tasks', 'Cancelled Tasks', 'Rejected Tasks',
        'On-time Rate (D)', 
        'Avg Delivery Time of Delivered Orders', 
        'Cancellation Rate from Delivery Issues'
    ]
    
    # إزالة الأعمدة التي استخدمت لإنشاء 'Agent Name'
    pivot_df = pivot_df.drop(columns=['Courier First Name', 'Courier Last Name'], errors='ignore')
    
    # التأكد من وجود جميع الأعمدة النهائية قبل الترتيب
    pivot_df = pivot_df[[col for col in final_cols if col in pivot_df.columns]]
    
    return pivot_df

def style_performance_table(df):
    """
    تطبيق التنسيق الشرطي (Conditional Highlighting) على جدول الأداء.
    الأخضر = أداء جيد، الأحمر = أداء سيئ.
    """
    
    style_df = df.copy()
    
    # 1. تحديد الأعمدة الرقمية الرئيسية للتنسيق (التأكد من وجودها)
    ontime_col = 'On-time Rate (D)'
    cancellation_col = 'Cancellation Rate from Delivery Issues'
    delivery_time_col = 'Avg Delivery Time of Delivered Orders'
    tph_col = 'TPH (Tasks Per Valid Hour)'
    
    # الأعمدة الموجودة بالفعل
    present_cols = [col for col in [ontime_col, cancellation_col, delivery_time_col, tph_col] if col in style_df.columns]
    
    if not present_cols:
        return df # لا يمكن تطبيق التنسيق إذا لم تكن الأعمدة موجودة

    # 2. تحويل النسب (0-1) إلى نسب مئوية (0-100) للحساب والعرض
    if ontime_col in style_df.columns:
        style_df[ontime_col] = style_df[ontime_col] * 100
    if cancellation_col in style_df.columns:
        style_df[cancellation_col] = style_df[cancellation_col] * 100
    
    # 3. حساب المتوسطات للمقارنة
    avg_ontime = style_df[ontime_col].mean() if ontime_col in style_df.columns else 0
    avg_delivery_time = style_df[delivery_time_col].mean() if delivery_time_col in style_df.columns else 0
    avg_cancellation = style_df[cancellation_col].mean() if cancellation_col in style_df.columns else 0
    avg_tph = style_df[tph_col].mean() if tph_col in style_df.columns else 0
    
    # 4. حساسية التلوين بناءً على الثابت PERFORMANCE_THRESHOLD
    LOW_THRESHOLD = PERFORMANCE_THRESHOLD 
    HIGH_THRESHOLD = 1 / PERFORMANCE_THRESHOLD 
    
    def highlight_performance(s):
        """تطبيق التلوين على الأعمدة بناءً على المتوسط."""
        
        styles = [''] * len(s) 
        
        # مؤشرات يجب أن تزيد (كلما زادت كان أفضل)
        positive_kpis = {ontime_col: avg_ontime, tph_col: avg_tph}
        # مؤشرات يجب أن تنقص (كلما نقصت كان أفضل)
        negative_kpis = {delivery_time_col: avg_delivery_time, cancellation_col: avg_cancellation}

        for col, avg_val in positive_kpis.items():
            if col in style_df.columns and avg_val > 0: # نتحقق من وجود العمود وأن المتوسط ليس صفراً
                col_idx = style_df.columns.get_loc(col)
                if s[col] < (avg_val * LOW_THRESHOLD):
                     styles[col_idx] = 'background-color: #f8d7da; color: #721c24' # أحمر للسيئ
                else:
                     styles[col_idx] = 'background-color: #d4edda; color: #155724' # أخضر للجيد

        for col, avg_val in negative_kpis.items():
            if col in style_df.columns and avg_val > 0: # نتحقق من وجود العمود وأن المتوسط ليس صفراً
                col_idx = style_df.columns.get_loc(col)
                # شرط إضافي لمعدل الإلغاء: لا يعتبر سيئاً ما لم يكن هناك إلغاء فعلي (أعلى من 2%)
                is_cancellation_issue = col == cancellation_col and s[col] > 2
                
                if s[col] > (avg_val * HIGH_THRESHOLD) or is_cancellation_issue:
                     styles[col_idx] = 'background-color: #f8d7da; color: #721c24' # أحمر للسيئ
                else:
                     styles[col_idx] = 'background-color: #d4edda; color: #155724' # أخضر للجيد

        return styles

    # تنسيق الأرقام
    format_dict = {}
    
    # تنسيق النسب المئوية
    if ontime_col in style_df.columns: format_dict[ontime_col] = '{:.2f}%'
    if cancellation_col in style_df.columns: format_dict[cancellation_col] = '{:.2f}%'
    
    # تنسيق الأرقام العشرية الأخرى
    if delivery_time_col in style_df.columns: format_dict[delivery_time_col] = '{:.2f}'
    if tph_col in style_df.columns: format_dict[tph_col] = '{:.2f}'
    if 'Valid Online Time' in style_df.columns: format_dict['Valid Online Time'] = '{:.2f}'
    if 'Courier App Online Time' in style_df.columns: format_dict['Courier App Online Time'] = '{:.2f}'

    # تنسيق الأرقام الصحيحة
    if 'Delivered Tasks' in style_df.columns: format_dict['Delivered Tasks'] = '{:,.0f}'
    if 'Accepted Tasks' in style_df.columns: format_dict['Accepted Tasks'] = '{:,.0f}'
    if 'Cancelled Tasks' in style_df.columns: format_dict['Cancelled Tasks'] = '{:,.0f}'
    if 'Rejected Tasks' in style_df.columns: format_dict['Rejected Tasks'] = '{:,.0f}'


    # تطبيق التنسيق على الجدول كله باستخدام Styler
    styled_df = style_df.style.apply(
        highlight_performance,
        axis=1, # تطبيق التلوين صف بصف
    ).format(format_dict)
    
    return styled_df


def analyze_performance(pivot_df):
    """
    تطبيق منطق العمل لإنشاء توصيات بناءً على المقارنة بالمتوسط.
    ** تم التعديل ليكون مرناً بناءً على الأعمدة المتاحة **
    """
    recommendations = {}

    analysis_df = pivot_df.copy()
    
    # أسماء الأعمدة المستخدمة في التحليل
    ontime_col = 'On-time Rate (D)'
    cancellation_col = 'Cancellation Rate from Delivery Issues'
    delivery_time_col = 'Avg Delivery Time of Delivered Orders'
    tph_col = 'TPH (Tasks Per Valid Hour)'
    valid_online_col = 'Valid Online Time'
    
    # التأكد من وجود الأعمدة اللازمة
    if tph_col not in analysis_df.columns: return {} # لا يمكن التحليل بدون TPH على الأقل

    # حساب المتوسطات للمقارنة
    avg_ontime = analysis_df[ontime_col].mean() if ontime_col in analysis_df.columns else 0
    avg_delivery_time = analysis_df[delivery_time_col].mean() if delivery_time_col in analysis_df.columns else 0
    avg_cancellation = analysis_df[cancellation_col].mean() if cancellation_col in analysis_df.columns else 0
    avg_tph = analysis_df[tph_col].mean()

    # استخدام القيمة الثابتة للحساسية
    LOW_PERFORMANCE_THRESHOLD = PERFORMANCE_THRESHOLD 
    HIGH_PERFORMANCE_THRESHOLD = 1 / PERFORMANCE_THRESHOLD 

    for index, row in analysis_df.iterrows():
        agent_name = row['Agent Name']
        notes = []

        # 1. تحليل الإنتاجية (Tasks Per Valid Hour)
        has_valid_time = valid_online_col in row and row[valid_online_col] > 5
        if row[tph_col] < (avg_tph * LOW_PERFORMANCE_THRESHOLD) and has_valid_time: # نراجع فقط من عمل أكثر من 5 ساعات
            notes.append(f"**📉 إنتاجية منخفضة (TPH):** يحقق {row[tph_col]:.2f} طلب/ساعة (أقل من متوسط الفريق). **التوصية:** توجيهه للعمل في أوقات الذروة ومراجعة منطق قبول الطلبات لتقليل فترة الانتظار.")
            
        # 2. تحليل كفاءة التسليم والالتزام بالوقت
        if ontime_col in analysis_df.columns and row[ontime_col] < (avg_ontime * LOW_PERFORMANCE_THRESHOLD) and avg_ontime > 0:
            notes.append(f"**🔴 انخفاض الالتزام بالوقت:** معدله {row[ontime_col]*100:.2f}% (أقل من متوسط الفريق). **التوصية:** تدريب على إدارة المسارات والبدء في الحركة بمجرد تأكيد الطلب لتجنب التأخير.")
        
        # 3. تحليل سرعة التسليم
        if delivery_time_col in analysis_df.columns and row[delivery_time_col] > (avg_delivery_time * HIGH_PERFORMANCE_THRESHOLD) and avg_delivery_time > 0:
            notes.append(f"**🟡 ارتفاع متوسط وقت التسليم:** متوسطه {row[delivery_time_col]:.2f} دقيقة (أبطأ من المتوسط). **التوصية:** التركيز على سرعة استلام الطلبات وتقليل وقت الانتظار في المطعم.")

        # 4. تحليل معدل الإلغاء
        if cancellation_col in analysis_df.columns and row[cancellation_col] > (avg_cancellation * HIGH_PERFORMANCE_THRESHOLD) and row[cancellation_col] * 100 > 2 and avg_cancellation > 0:
            notes.append(f"**❌ معدل إلغاء مرتفع:** معدله {row[cancellation_col]*100:.2f}%. **التوصية:** التحقيق الفوري في سبب الإلغاءات المتكررة (مشاكل تحديد الموقع/التواصل مع العميل/أخطاء النظام).")


        # تجميع الملاحظات
        if notes:
            recommendations[agent_name] = {'ID': row['Courier ID'], 'Notes': notes}

    return recommendations

def to_excel(df):
    """دالة تحويل DataFrame إلى ملف Excel في الذاكرة لتمكين التصدير."""
    output = BytesIO()
    
    export_df = df.copy()
    
    # الأعمدة التي يجب تحويلها إلى نسبة مئوية بـ %
    percent_cols = ['On-time Rate (D)', 'Cancellation Rate from Delivery Issues']
    
    # تحديد الأعمدة التي سيتم تحويلها (الموجودة في DataFrame)
    cols_to_convert = [col for col in percent_cols if col in export_df.columns]
    
    for col in cols_to_convert:
        export_df[col + ' (%)'] = (export_df.pop(col) * 100).round(2)
    
    # ترتيب الأعمدة للتصدير
    final_cols_order = [
        'Courier ID', 'Agent Name', 
        'Valid Online Time', 'Courier App Online Time',
        'TPH (Tasks Per Valid Hour)',
        'Delivered Tasks', 'Accepted Tasks', 'Cancelled Tasks', 'Rejected Tasks',
        'On-time Rate (D) (%)', 
        'Avg Delivery Time of Delivered Orders', 
        'Cancellation Rate from Delivery Issues (%)'
    ]
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # التأكد من تصدير الأعمدة الموجودة فقط بالترتيب المطلوب
        cols_to_export = [col for col in final_cols_order if col in export_df.columns]
        export_df[cols_to_export].to_excel(writer, index=False, sheet_name='Keeta_Delivery_Report_Summary')
            
    processed_data = output.getvalue()
    return processed_data

# ==============================================================================
# 3. واجهة التطبيق الرئيسية (Streamlit)
# ==============================================================================

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="أداة تحليل أداء مناديب كيتا")
st.title("🛵 محلل أداء مناديب التوصيل المتقدم (كيتا)")
st.markdown("---")
st.markdown("✅ **تم التعديل:** تم إصلاح خطأ `SyntaxError` **وزيادة مرونة** تحديد الأعمدة ليعمل على أي ملف إكسيل مشابه.")

# تحديد عتبة الحساسية في الواجهة للسماح للمستخدم بتغييرها (ميزة إضافية)
st.sidebar.header("إعدادات التحليل")
sensitivity_slider = st.sidebar.slider(
    'عتبة الحساسية (تحت المتوسط):', 
    min_value=0.5, max_value=1.0, value=PERFORMANCE_THRESHOLD, step=0.05,
    help="إذا كان أداء المندوب أقل من هذه النسبة من متوسط الفريق، يعتبر أداء سيئاً (مثال: 0.90 يعني أقل بـ 10%)"
)
# تحديث الثابت العالمي بناءً على اختيار المستخدم
PERFORMANCE_THRESHOLD = sensitivity_slider
st.sidebar.info(f"التحليل يستخدم عتبة **{int(sensitivity_slider*100)}%**")


# **التعديل الجديد:** استخدام st.file_uploader لتمكين التحميل المحلي.
uploaded_file = st.file_uploader("📥 **يرجى رفع ملف الإكسيل/CSV الخاص ببيانات المناديب**", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # قراءة البيانات مع تحديد نوع الملف
        if uploaded_file.name.endswith('.csv'):
             # قراءة ملف CSV
             df = pd.read_csv(uploaded_file)
        else:
             # قراءة ملف Excel
             df = pd.read_excel(uploaded_file)
        
        # 1. تنظيف ومعالجة البيانات
        initial_count = len(df)
        df = clean_and_process_data(df)
        
        filtered_count = initial_count - len(df)
        st.success(f"تم تحميل الملف **{uploaded_file.name}** بنجاح. تم استبعاد **{filtered_count}** سجل (لعدم وجود ساعات عمل فعالة).")
        
        # عرض البيانات الأولية
        st.subheader("📋 نموذج البيانات بعد المعالجة (أول 5 من السجلات الفعالة)")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        st.markdown("---")

        # ==================================================
        # 2. إنشاء وعرض الجدول المحوري المنسق
        # ==================================================
        
        st.header("📈 تقرير أداء المناديب المجمّع (مُنسَّق)")
        pivot_table = generate_pivot_table(df)
        
        # تطبيق التنسيق الشرطي (Highlighting)
        styled_table = style_performance_table(pivot_table)
        
        # عرض الجدول المحوري المنسق
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: small;'>
            **مفتاح الألوان:**<br>
            <span style='color: #155724;'>■ الأخضر:</span> أداء المندوب جيد (أفضل من عتبة الـ {int(PERFORMANCE_THRESHOLD*100)}% من متوسط الفريق).<br>
            <span style='color: #721c24;'>■ الأحمر:</span> أداء المندوب سيئ (أقل من عتبة الـ {int(PERFORMANCE_THRESHOLD*100)}% من متوسط الفريق).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")


        # زر تصدير الإكسيل
        st.download_button(
            label="⬇️ اضغط للتصدير كملف Excel (ملخص الأداء)",
            data=to_excel(pivot_table),
            file_name="Keeta_Delivery_Performance_Summary_Report.xlsx",
            mime="application/vnd.ms-excel"
        )

        st.markdown("---")

        # ==================================================
        # 3. عرض التوصيات والتحليل
        # ==================================================
        
        st.header("📝 التوصيات ونوتات الأداء السيئ")
        recommendations = analyze_performance(pivot_table)

        if recommendations:
            st.warning(f"⚠️ **تنبيه:** تم تحديد **{len(recommendations)}** من المناديب بأداء أقل من العتبة المحددة ({int(PERFORMANCE_THRESHOLD*100)}%)، ويحتاجون إلى مراجعة:")
            
            # عرض التوصيات
            for agent, data in recommendations.items():
                st.markdown(f"### 👤 المندوب: {agent} (ID: {data['ID']})")
                for note in data['Notes']:
                    st.markdown(f"- {note}")
                st.markdown("---")
        else:
            st.balloons()
            st.success("🎉 **أداء ممتاز!** جميع المناديب ضمن الحدود المقبولة ولا يحتاجون إلى توصيات فورية.")

    except ValueError as ve:
        st.error(f"❌ خطأ في هيكل الملف: {ve}")
        st.markdown("يرجى التأكد من أن الملف يحتوي على جميع الأعمدة المطلوبة بالأسماء الصحيحة والمطابقة لملف الإكسيل الأصلي.")
    except Exception as e:
        st.error(f"❌ حدث خطأ غير متوقع أثناء المعالجة: {e}")
        st.markdown("**نصيحة:** قد يكون هناك مشكلة في تنسيق البيانات داخل الملف أو في الأعمدة المحفوظة.")
else:
    st.info("الرجاء رفع ملف الإكسيل أو CSV للبدء في تحليل أداء المناديب.")
