import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
import re

# ==============================================================================
# 1. تحديد الثوابت وخريطة الأعمدة المطلوبة
# ==============================================================================

# *** التحكم في حساسية التلوين والتوصيات ***
PERFORMANCE_THRESHOLD = 0.90 

# خريطة الأعمدة المطلوبة بالاسم الداخلي (المستخدم في الكود) والاسم العربي (للعرض في الواجهة)
REQUIRED_KPI_MAPPING = {
    'Courier ID': 'هوية المندوب (ID)',
    'Agent Name': 'الاسم الكامل للمندوب', # هذا سيتم إنشاؤه لاحقاً
    'Valid Online Time': 'ساعات العمل الفعالة (ضروري)', 
    'On-time Rate (D)': 'معدل الالتزام بالوقت (ضروري)',
    'Cancellation Rate from Delivery Issues': 'معدل الإلغاء (مشاكل التسليم) (ضروري)',
    'Courier App Online Time': 'وقت الاتصال بالتطبيق',
    'Accepted Tasks': 'الطلبات المقبولة',
    'Delivered Tasks': 'الطلبات المسلّمة',
    'Cancelled Tasks': 'الطلبات الملغاة',
    'Rejected Tasks': 'الطلبات المرفوضة',
    'Avg Delivery Time of Delivered Orders': 'متوسط وقت التسليم'
}

# أسماء الأعمدة الحاسمة التي يجب توافرها لبدء التحليل
CRITICAL_COLS = ['Courier ID', 'Valid Online Time', 'On-time Rate (D)', 'Cancellation Rate from Delivery Issues', 'Delivered Tasks']

# مفاتيح البحث التلقائي المرنة (للتخمين الأولي)
FLEXIBLE_SEARCH_KEYS = {
    'id': 'Courier ID',
    'valid': 'Valid Online Time',
    'ontime': 'On-time Rate (D)',
    'cancel': 'Cancellation Rate from Delivery Issues',
    'accepted': 'Accepted Tasks',
    'delivered': 'Delivered Tasks',
    'deliverytime': 'Avg Delivery Time of Delivered Orders',
    'apponline': 'Courier App Online Time',
    'first name': 'Courier First Name',
    'last name': 'Courier Last Name',
    
    # 🌟 تحسينات لدعم أسماء الأعمدة التي قدمتها (Cancelled Tasks, Rejected Tasks)
    'cancelled': 'Cancelled Tasks',
    'rejected': 'Rejected Tasks',
}


# ==============================================================================
# 2. الدوال المساعدة
# ==============================================================================

def guess_column(required_key, available_cols):
    """يخمن اسم العمود من قائمة الأعمدة المتاحة بناءً على مفاتيح البحث المرنة."""
    required_key_lower = required_key.lower().replace(' ', '')

    # 1. محاولة المطابقة التامة مع الأسماء الداخلية (إذا كان المستخدم يستخدم تقاريرنا)
    if required_key in available_cols:
        return required_key
    
    # 2. محاولة المطابقة المرنة (جزء من الاسم)
    for key_fragment, internal_name in FLEXIBLE_SEARCH_KEYS.items():
        if internal_name == required_key:
            for col in available_cols:
                # تنظيف الاسم المتاح للمقارنة
                normalized_col = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
                
                # البحث عن المفتاح الجزئي داخل الاسم الموحد
                if key_fragment.lower().replace(' ', '') in normalized_col:
                    return col
    
    return '(لم يتم الاختيار)'


def clean_and_process_data(df, user_map):
    """
    تنظيف وإعادة تسمية الأعمدة بناءً على مطابقة المستخدم.
    """
    
    # 1. تطبيق خريطة المستخدم لإعادة تسمية الأعمدة
    # نستخدم user_map كـ {الاسم_الداخلي: الاسم_المحدد_في_الملف}
    # ونقوم بعكسها لـ {الاسم_المحدد_في_الملف: الاسم_الداخلي}
    reverse_map = {v: k for k, v in user_map.items()}
    
    # تصفية DataFrame للاحتفاظ فقط بالأعمدة التي اختارها المستخدم
    df = df[[col for col in reverse_map.keys() if col in df.columns]].rename(columns=reverse_map)

    # 2. تحويل البيانات إلى أرقام (باستخدام الأسماء الداخلية الجديدة)
    numeric_cols = [
        'Courier App Online Time', 'Valid Online Time', 'Accepted Tasks', 
        'Delivered Tasks', 'Cancelled Tasks', 'Rejected Tasks', 
        'On-time Rate (D)', 'Avg Delivery Time of Delivered Orders', 
        'Cancellation Rate from Delivery Issues'
    ]

    for col in numeric_cols:
        if col in df.columns:
            # تحويل القيم التي قد تكون في شكل سلاسل نصية أو تحتوي على رموز (%) إلى أرقام
            df[col] = df[col].astype(str).str.replace('[^0-9.+-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. دمج الاسم (إذا لم يكن موجوداً)
    if 'Agent Name' not in df.columns and 'Courier First Name' in df.columns and 'Courier Last Name' in df.columns:
        df['Agent Name'] = df['Courier First Name'].fillna('') + ' ' + df['Courier Last Name'].fillna('')
    elif 'Agent Name' not in df.columns:
         # إذا لم يتوفر الاسم الأول والأخير، نحاول استخدام ID كاسم مؤقت
         df['Agent Name'] = 'Agent_' + df['Courier ID'].astype(str)
    
    # 4. تصفية الصفوف التي لا تحتوي على ID للمندوب
    if 'Courier ID' in df.columns:
        df = df.dropna(subset=['Courier ID'])
    
    # 5. 🔴 فلترة المناديب الذين لم يعملوا (ساعات الأونلاين الفعالة 0)
    if 'Valid Online Time' in df.columns:
        df = df[df['Valid Online Time'] > 0].reset_index(drop=True)
    
    return df

def generate_pivot_table(df):
    """ينشئ الجدول المحوري (Pivot Table) بتجميع مؤشرات الأداء المطلوبة."""
    
    group_cols = ['Courier ID', 'Agent Name']
    
    # الأعمدة المتاحة للحساب بناءً على ما تبقى بعد التنظيف
    available_cols = [col for col in df.columns if col not in group_cols and col not in ['Courier First Name', 'Courier Last Name']]
    
    # قاموس التجميع (Aggregation Dictionary) بناءً على الأعمدة المتاحة
    agg_dict = {}
    
    # تحديد وظيفة التجميع لكل عمود (جمع للكميات، متوسط للمعدلات والأوقات)
    for col in available_cols:
        if any(keyword in col for keyword in ['Time', 'Tasks', 'Cancelled', 'Rejected', 'Accepted', 'Delivered']):
            agg_dict[col] = 'sum'
        elif any(keyword in col for keyword in ['Rate', 'Avg']):
            agg_dict[col] = 'mean'
            
    if not agg_dict:
        return pd.DataFrame()
        
    pivot_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # 🌟 إضافة مؤشر TPH (الإنتاجية) كأهم مؤشر جديد
    if 'Delivered Tasks' in pivot_df.columns and 'Valid Online Time' in pivot_df.columns:
        pivot_df['TPH (Tasks Per Valid Hour)'] = np.where(
            pivot_df['Valid Online Time'] > 0,
            (pivot_df['Delivered Tasks'] / pivot_df['Valid Online Time']),
            0
        ).round(2)
    else:
        pivot_df['TPH (Tasks Per Valid Hour)'] = 0

    
    # ترتيب الأعمدة للعرض النهائي (باستخدام الأسماء الداخلية)
    internal_cols_order = list(REQUIRED_KPI_MAPPING.keys())
    internal_cols_order.insert(4, 'TPH (Tasks Per Valid Hour)') # إضافة TPH بعد Valid Online Time
    
    # تصفية وترتيب الأعمدة الموجودة بالفعل
    pivot_df = pivot_df[[col for col in internal_cols_order if col in pivot_df.columns]]
    
    return pivot_df

# (دوال style_performance_table و analyze_performance و to_excel تبقى كما هي تقريباً، مع التأكد من استخدام الثوابت والقيم المحسوبة)
# ملاحظة: تم تعديل style_performance_table و analyze_performance في التعديل السابق لتعمل على الأعمدة الداخلية (مثل On-time Rate (D))، لذا سأبقيها كما هي.

def style_performance_table(df):
    """تطبيق التنسيق الشرطي (Conditional Highlighting) على جدول الأداء."""
    
    style_df = df.copy()
    current_threshold = st.session_state.performance_threshold # استخدام القيمة المحدثة من الواجهة
    
    # 1. تحديد الأعمدة الرقمية الرئيسية للتنسيق (التأكد من وجودها)
    ontime_col = 'On-time Rate (D)'
    cancellation_col = 'Cancellation Rate from Delivery Issues'
    delivery_time_col = 'Avg Delivery Time of Delivered Orders'
    tph_col = 'TPH (Tasks Per Valid Hour)'
    
    present_cols = [col for col in [ontime_col, cancellation_col, delivery_time_col, tph_col] if col in style_df.columns]
    
    if not present_cols:
        return df

    # 2. تحويل النسب (0-1) إلى نسب مئوية (0-100) للحساب والعرض
    for col in [ontime_col, cancellation_col]:
        if col in style_df.columns:
            style_df[col] = style_df[col] * 100
    
    # 3. حساب المتوسطات للمقارنة
    avg_metrics = {col: style_df[col].mean() for col in present_cols}
    
    # 4. حساسية التلوين بناءً على الثابت PERFORMANCE_THRESHOLD
    LOW_THRESHOLD = current_threshold
    HIGH_THRESHOLD = 1 / current_threshold
    
    def highlight_performance(s):
        styles = [''] * len(s) 
        
        # مؤشرات يجب أن تزيد (كلما زادت كان أفضل)
        positive_kpis = {ontime_col, tph_col}
        # مؤشرات يجب أن تنقص (كلما نقصت كان أفضل)
        negative_kpis = {delivery_time_col, cancellation_col}

        for i, col in enumerate(style_df.columns):
            if col in positive_kpis and col in avg_metrics and avg_metrics[col] > 0:
                if s[col] < (avg_metrics[col] * LOW_THRESHOLD):
                     styles[i] = 'background-color: #f8d7da; color: #721c24'
                else:
                     styles[i] = 'background-color: #d4edda; color: #155724'

            elif col in negative_kpis and col in avg_metrics and avg_metrics[col] > 0:
                is_cancellation_issue = col == cancellation_col and s[col] > 2
                
                if s[col] > (avg_metrics[col] * HIGH_THRESHOLD) or is_cancellation_issue:
                     styles[i] = 'background-color: #f8d7da; color: #721c24'
                else:
                     styles[i] = 'background-color: #d4edda; color: #155724'
        return styles

    # تنسيق الأرقام
    format_dict = {}
    for col in [ontime_col, cancellation_col]:
        if col in style_df.columns: format_dict[col] = '{:.2f}%'
    for col in [delivery_time_col, tph_col, 'Valid Online Time', 'Courier App Online Time']:
        if col in style_df.columns: format_dict[col] = '{:.2f}'
    for col in ['Delivered Tasks', 'Accepted Tasks', 'Cancelled Tasks', 'Rejected Tasks']:
        if col in style_df.columns: format_dict[col] = '{:,.0f}'

    styled_df = style_df.style.apply(highlight_performance, axis=1).format(format_dict)
    
    # إرجاع الأسماء العربية للعرض
    arabic_map_display = {k: v for k, v in REQUIRED_KPI_MAPPING.items() if k in df.columns}
    styled_df.columns = [arabic_map_display.get(col, col) for col in styled_df.columns]
    
    return styled_df

def analyze_performance(pivot_df):
    """تطبيق منطق العمل لإنشاء توصيات بناءً على المقارنة بالمتوسط."""
    recommendations = {}

    analysis_df = pivot_df.copy()
    current_threshold = st.session_state.performance_threshold

    # أسماء الأعمدة المستخدمة في التحليل
    ontime_col = 'On-time Rate (D)'
    cancellation_col = 'Cancellation Rate from Delivery Issues'
    delivery_time_col = 'Avg Delivery Time of Delivered Orders'
    tph_col = 'TPH (Tasks Per Valid Hour)'
    valid_online_col = 'Valid Online Time'
    
    # التأكد من وجود الأعمدة اللازمة
    required_for_analysis = [col for col in [tph_col, ontime_col, cancellation_col, delivery_time_col] if col in analysis_df.columns]
    if not required_for_analysis: return {}

    # حساب المتوسطات للمقارنة
    avg_metrics = {col: analysis_df[col].mean() for col in required_for_analysis}
    
    LOW_PERFORMANCE_THRESHOLD = current_threshold 
    HIGH_PERFORMANCE_THRESHOLD = 1 / current_threshold 

    for index, row in analysis_df.iterrows():
        agent_name = row['Agent Name']
        notes = []

        # 1. تحليل الإنتاجية (TPH)
        has_valid_time = valid_online_col in row and row[valid_online_col] > 5
        if tph_col in row and row[tph_col] < (avg_metrics.get(tph_col, 0) * LOW_PERFORMANCE_THRESHOLD) and has_valid_time:
            notes.append(f"**📉 إنتاجية منخفضة (TPH):** يحقق {row[tph_col]:.2f} طلب/ساعة. **التوصية:** مراجعة منطق قبول الطلبات لتقليل فترة الانتظار.")
            
        # 2. تحليل كفاءة التسليم والالتزام بالوقت
        if ontime_col in row and row[ontime_col] < (avg_metrics.get(ontime_col, 0) * LOW_PERFORMANCE_THRESHOLD) and avg_metrics.get(ontime_col, 0) > 0:
            notes.append(f"**🔴 انخفاض الالتزام بالوقت:** معدله {row[ontime_col]*100:.2f}%. **التوصية:** تدريب على إدارة المسارات لتجنب التأخير.")
        
        # 3. تحليل سرعة التسليم
        if delivery_time_col in row and row[delivery_time_col] > (avg_metrics.get(delivery_time_col, 0) * HIGH_PERFORMANCE_THRESHOLD) and avg_metrics.get(delivery_time_col, 0) > 0:
            notes.append(f"**🟡 ارتفاع متوسط وقت التسليم:** متوسطه {row[delivery_time_col]:.2f} دقيقة. **التوصية:** التركيز على سرعة استلام الطلبات وتقليل وقت الانتظار.")

        # 4. تحليل معدل الإلغاء
        if cancellation_col in row and row[cancellation_col] > (avg_metrics.get(cancellation_col, 0) * HIGH_PERFORMANCE_THRESHOLD) and row[cancellation_col] * 100 > 2 and avg_metrics.get(cancellation_col, 0) > 0:
            notes.append(f"**❌ معدل إلغاء مرتفع:** معدله {row[cancellation_col]*100:.2f}%. **التوصية:** التحقيق الفوري في سبب الإلغاءات المتكررة (مشاكل تحديد الموقع/التواصل).")


        # تجميع الملاحظات
        if notes:
            recommendations[agent_name] = {'ID': row['Courier ID'], 'Notes': notes}

    return recommendations

def to_excel(df):
    """دالة تحويل DataFrame إلى ملف Excel في الذاكرة لتمكين التصدير."""
    output = BytesIO()
    
    export_df = df.copy()
    
    # تحويل النسب
    percent_cols = ['On-time Rate (D)', 'Cancellation Rate from Delivery Issues']
    cols_to_convert = [col for col in percent_cols if col in export_df.columns]
    
    for col in cols_to_convert:
        export_df[col + ' (%)'] = (export_df.pop(col) * 100).round(2)
    
    # استخدام خريطة الأسماء العربية للتصدير
    arabic_map_export = {k: REQUIRED_KPI_MAPPING.get(k, k) for k in export_df.columns}
    arabic_map_export['TPH (Tasks Per Valid Hour)'] = 'الإنتاجية (TPH)'
    export_df.columns = [arabic_map_export.get(col, col) for col in export_df.columns]
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Keeta_Delivery_Report_Summary')
            
    processed_data = output.getvalue()
    return processed_data

# ==============================================================================
# 3. واجهة التطبيق الرئيسية (Streamlit)
# ==============================================================================

# إعداد الصفحة وحالة الجلسة
st.set_page_config(layout="wide", page_title="أداة تحليل أداء مناديب كيتا")
if 'performance_threshold' not in st.session_state:
    st.session_state.performance_threshold = PERFORMANCE_THRESHOLD

st.title("🛵 محلل أداء مناديب التوصيل المتقدم (كيتا)")
st.markdown("---")
st.markdown("✅ **تم التعديل الرئيسي:** تم إضافة **أداة مطابقة الأعمدة اليدوية** في الشريط الجانبي لضمان عمل التحليل على أي ملف ترفعه.")

# تحديد عتبة الحساسية
st.sidebar.header("إعدادات التحليل")
sensitivity_slider = st.sidebar.slider(
    'عتبة الحساسية (تحت المتوسط):', 
    min_value=0.5, max_value=1.0, value=st.session_state.performance_threshold, step=0.05,
    help="إذا كان أداء المندوب أقل من هذه النسبة من متوسط الفريق، يعتبر أداء سيئاً."
)
st.session_state.performance_threshold = sensitivity_slider
st.sidebar.info(f"التحليل يستخدم عتبة **{int(sensitivity_slider*100)}%**")


# واجهة رفع الملف
uploaded_file = st.file_uploader("📥 **الخطوة 1: يرجى رفع ملف الإكسيل/CSV الخاص ببيانات المناديب**", type=["xlsx", "xls", "csv"])

# متغير لتخزين خريطة الأعمدة التي يختارها المستخدم
user_map = {}
data_frame = None

if uploaded_file is not None:
    try:
        # قراءة البيانات
        if uploaded_file.name.endswith('.csv'):
             data_frame = pd.read_csv(uploaded_file)
        else:
             data_frame = pd.read_excel(uploaded_file)
        
        data_frame.columns = data_frame.columns.astype(str).str.strip()
        available_cols = data_frame.columns.tolist()
        available_cols_options = ['(لم يتم الاختيار)'] + available_cols

        # ==================================================
        # 🌟 الخطوة 2: عرض واجهة مطابقة الأعمدة 🌟
        # ==================================================
        
        st.sidebar.header("📝 الخطوة 2: مطابقة الأعمدة يدوياً")
        st.sidebar.markdown("يرجى اختيار اسم العمود المقابل في ملفك لكل متطلب:")

        for required_internal_name, arabic_label in REQUIRED_KPI_MAPPING.items():
            # إذا كان الاسم الكامل مكوناً من اسمين (Agent Name)، يتم إنشاؤه لاحقاً
            if required_internal_name == 'Agent Name':
                continue
            
            # محاولة التخمين الأولي
            guessed_col = guess_column(required_internal_name, available_cols)
            
            # تحديد الـ index الذي يجب أن يظهر عليه التخمين (إذا كان موجوداً)
            initial_index = available_cols.index(guessed_col) + 1 if guessed_col in available_cols else 0
            
            # عرض قائمة الاختيار
            selected_col = st.sidebar.selectbox(
                f"**{arabic_label}**",
                available_cols_options,
                index=initial_index,
                key=f'map_{required_internal_name}'
            )
            
            # إضافة الاختيار إلى خريطة المستخدم
            if selected_col != '(لم يتم الاختيار)':
                user_map[required_internal_name] = selected_col

        
        # --------------------------------------------------
        # 🌟 الخطوة 3: التحقق وبدء التحليل
        # --------------------------------------------------
        
        # التحقق من أن الأعمدة الحاسمة تم اختيارها يدوياً
        mapped_critical_cols = [col for col in CRITICAL_COLS if col in user_map]

        if len(mapped_critical_cols) < len(CRITICAL_COLS):
            missing = [REQUIRED_KPI_MAPPING[col] for col in CRITICAL_COLS if col not in user_map]
            st.error(f"❌ **توقف التحليل:** يجب تحديد الأعمدة الأساسية التالية في الشريط الجانبي لبدء التحليل: {', '.join(missing)}")
            st.warning("يرجى الانتقال إلى الشريط الجانبي (إذا لم يظهر، اضغط على السهم > في أعلى يسار الشاشة) وإكمال مطابقة الأعمدة.")
        else:
            # إذا تم تحديد جميع الأعمدة الحاسمة
            st.success("✅ تم مطابقة الأعمدة الأساسية بنجاح. يتم الآن معالجة البيانات...")

            # 1. تنظيف ومعالجة البيانات باستخدام خريطة المستخدم
            initial_count = len(data_frame)
            processed_df = clean_and_process_data(data_frame.copy(), user_map)
            
            filtered_count = initial_count - len(processed_df)
            st.success(f"تم تحميل الملف بنجاح. تم استبعاد **{filtered_count}** سجل (لعدم وجود ساعات عمل فعالة).")
            
            st.subheader("📋 نموذج البيانات بعد المعالجة")
            st.dataframe(processed_df.head(), use_container_width=True, hide_index=True)
            st.markdown("---")

            # ==================================================
            # 2. إنشاء وعرض الجدول المحوري المنسق
            # ==================================================
            
            st.header("📈 تقرير أداء المناديب المجمّع (مُنسَّق)")
            pivot_table = generate_pivot_table(processed_df)
            
            # تطبيق التنسيق الشرطي (Highlighting)
            styled_table = style_performance_table(pivot_table)
            
            # عرض الجدول المحوري المنسق
            st.dataframe(styled_table, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: small;'>
                **مفتاح الألوان:**<br>
                <span style='color: #155724;'>■ الأخضر:</span> أداء المندوب جيد (أفضل من عتبة الـ {int(st.session_state.performance_threshold*100)}% من متوسط الفريق).<br>
                <span style='color: #721c24;'>■ الأحمر:</span> أداء المندوب سيئ (أقل من عتبة الـ {int(st.session_state.performance_threshold*100)}% من متوسط الفريق).
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
                st.warning(f"⚠️ **تنبيه:** تم تحديد **{len(recommendations)}** من المناديب بأداء أقل من العتبة المحددة، ويحتاجون إلى مراجعة:")
                
                for agent, data in recommendations.items():
                    st.markdown(f"### 👤 المندوب: {agent} (ID: {data['ID']})")
                    for note in data['Notes']:
                        st.markdown(f"- {note}")
                    st.markdown("---")
            else:
                st.balloons()
                st.success("🎉 **أداء ممتاز!** جميع المناديب ضمن الحدود المقبولة ولا يحتاجون إلى توصيات فورية.")

    except Exception as e:
        st.error(f"❌ حدث خطأ غير متوقع أثناء المعالجة: {e}")
        st.markdown("**نصيحة:** يرجى التأكد من أن الأعمدة التي اخترتها تحتوي على بيانات رقمية وليست نصوصاً غير قابلة للتحويل.")
else:
    st.info("الرجاء رفع ملف الإكسيل أو CSV للبدء في تحليل أداء المناديب.")
