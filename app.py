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

# قائمة الأعمدة القياسية التي يتم استخدامها في التحليل
STANDARD_COLS = {
    'Courier ID': 'ID',
    'Courier First Name': 'First Name',
    'Courier Last Name': 'Last Name',
    'Valid Online Time': 'Online Time (h)',
    'Delivered Tasks': 'Delivered Tasks',
    'On-time Rate (D)': 'On-time Rate', # معدل الالتزام
    'Avg Delivery Time of Delivered Orders': 'Avg Delivery Time (min)', # متوسط وقت التسليم
    'Cancellation Rate from Delivery Issues': 'Cancellation Rate' # معدل الإلغاء
}

# ==============================================================================
# 2. الدوال المساعدة لتحميل ومعالجة البيانات
# ==============================================================================

def clean_and_process_data(df):
    """
    تنظيف وتوحيد أسماء الأعمدة وتحويل البيانات للتحليل.
    تتم تسمية الأعمدة بناءً على مفاتيح القاموس STANDARD_COLS
    """
    
    # تنظيف أسماء الأعمدة من المسافات الزائدة وإزالة أي رموز غير مرغوب فيها
    df.columns = df.columns.str.strip().str.replace('[^a-zA-Z0-9\s-]', '', regex=True)
    
    # إعادة تسمية الأعمدة الموجودة في الملف إلى الأسماء القياسية للتحليل
    current_cols = {old: new for old, new in STANDARD_COLS.items() if old in df.columns}
    df = df.rename(columns=current_cols, errors='ignore')

    # التأكد من وجود الأعمدة الأساسية اللازمة للتحليل
    if 'ID' not in df.columns or 'Online Time (h)' not in df.columns:
        raise ValueError("الملف لا يحتوي على الأعمدة الأساسية المطلوبة: 'Courier ID' و 'Valid Online Time'.")

    # التأكد من تحويل الأعمدة الرقمية إلى النوع float
    for col in ['Online Time (h)', 'Delivered Tasks', 'On-time Rate', 'Avg Delivery Time (min)', 'Cancellation Rate']:
        if col in df.columns:
            # تحويل القيم التي قد تكون في شكل سلاسل نصية (مثل 30.5h) إلى أرقام
            df[col] = df[col].astype(str).str.replace('[^0-9.+-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # تصفية الصفوف التي لا تحتوي على ID للمندوب أو لا يوجد بها ساعات عمل (المطلب الجديد)
    df = df.dropna(subset=['ID'])
    # 🔴 الميزة المضافة: فلترة المناديب الذين لم يعملوا (ساعات الأونلاين 0)
    df = df[df['Online Time (h)'] > 0].reset_index(drop=True)
    
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
    
    # حساب الإنتاجية (Tasks Per Hour)
    pivot_df['Tasks Per Hour'] = np.where(
        pivot_df['Total_Online_Hours'] > 0,
        (pivot_df['Total_Delivered_Tasks'] / pivot_df['Total_Online_Hours']),
        0
    ).round(2)

    # إعادة تسمية الأعمدة لتكون باللغة العربية
    pivot_df = pivot_df.rename(columns={
        'ID': 'هوية المندوب (ID)',
        'Agent Name': 'اسم المندوب',
        'Total_Delivered_Tasks': 'الطلبات المنجزة',
        'Total_Online_Hours': 'إجمالي الساعات أونلاين',
        'Tasks Per Hour': 'الإنتاجية (TPH)',
        'Avg_On_time_Rate': 'معدل الالتزام (نسبة)', # نتركها كنسبة داخلية (0-1) للتنسيق والتلوين
        'Avg_Delivery_Time': 'متوسط وقت التسليم (دقيقة)',
        'Avg_Cancellation_Rate': 'معدل الإلغاء (نسبة)' # نتركها كنسبة داخلية (0-1) للتنسيق والتلوين
    })
    
    # ترتيب الأعمدة للعرض النهائي
    final_cols = ['هوية المندوب (ID)', 'اسم المندوب', 'الطلبات المنجزة', 'إجمالي الساعات أونلاين', 'الإنتاجية (TPH)',
                  'معدل الالتزام (نسبة)', 'متوسط وقت التسليم (دقيقة)', 'معدل الإلغاء (نسبة)']
    
    pivot_df = pivot_df[[col for col in final_cols if col in pivot_df.columns]]
    
    return pivot_df

def style_performance_table(df):
    """
    تطبيق التنسيق الشرطي (Conditional Highlighting) على جدول الأداء.
    الأخضر = أداء جيد، الأحمر = أداء سيئ.
    """
    # نسخ الجدول لتحويل النسب إلى أرقام (0-100) للتنسيق
    style_df = df.copy()
    
    # تحويل النسب الداخلية (0-1) إلى نسب مئوية (0-100) للعرض
    style_df['معدل الالتزام (نسبة)'] = style_df['معدل الالتزام (نسبة)'] * 100
    style_df['معدل الإلغاء (نسبة)'] = style_df['معدل الإلغاء (نسبة)'] * 100
    
    # حساب المتوسطات للمقارنة
    avg_ontime = style_df['معدل الالتزام (نسبة)'].mean()
    avg_delivery_time = style_df['متوسط وقت التسليم (دقيقة)'].mean()
    avg_cancellation = style_df['معدل الإلغاء (نسبة)'].mean()
    avg_tph = style_df['الإنتاجية (TPH)'].mean()
    
    # حساسية التلوين بناءً على الثابت PERFORMANCE_THRESHOLD
    LOW_THRESHOLD = PERFORMANCE_THRESHOLD 
    HIGH_THRESHOLD = 1 / PERFORMANCE_THRESHOLD 
    
    def highlight_performance(s):
        """تطبيق التلوين على الأعمدة بناءً على المتوسط."""
        
        # مؤشرات أفضل بالزيادة (On-time Rate, TPH)
        is_worst_positive = s[['معدل الالتزام (نسبة)', 'الإنتاجية (TPH)']] < [avg_ontime * LOW_THRESHOLD, avg_tph * LOW_THRESHOLD]
        
        # مؤشرات أفضل بالنقصان (Delivery Time, Cancellation Rate)
        is_worst_negative = s[['متوسط وقت التسليم (دقيقة)', 'معدل الإلغاء (نسبة)']] > [avg_delivery_time * HIGH_THRESHOLD, avg_cancellation * HIGH_THRESHOLD]

        styles = [''] * len(s) 
        
        # تحديد موقع الأعمدة
        try:
            ontime_idx = style_df.columns.get_loc('معدل الالتزام (نسبة)')
            tph_idx = style_df.columns.get_loc('الإنتاجية (TPH)')
            delivery_time_idx = style_df.columns.get_loc('متوسط وقت التسليم (دقيقة)')
            cancellation_idx = style_df.columns.get_loc('معدل الإلغاء (نسبة)')
            
            # 1. معدل الالتزام (%)
            if is_worst_positive[0]:
                 styles[ontime_idx] = 'background-color: #f8d7da; color: #721c24' # أحمر للسيئ
            else:
                 styles[ontime_idx] = 'background-color: #d4edda; color: #155724' # أخضر للجيد

            # 2. الإنتاجية (TPH)
            if is_worst_positive[1]:
                 styles[tph_idx] = 'background-color: #f8d7da; color: #721c24'
            else:
                 styles[tph_idx] = 'background-color: #d4edda; color: #155724'

            # 3. وقت التسليم (دقيقة)
            if is_worst_negative[0]:
                 styles[delivery_time_idx] = 'background-color: #f8d7da; color: #721c24'
            else:
                 styles[delivery_time_idx] = 'background-color: #d4edda; color: #155724'
                
            # 4. معدل الإلغاء (%)
            # نضع حد إضافي لكي لا يظهر تلوين أحمر لمندوب لديه معدل إلغاء 0.01%
            if is_worst_negative[1] and s['معدل الإلغاء (نسبة)'] > 2: # معدل إلغاء فعلي فوق 2%
                 styles[cancellation_idx] = 'background-color: #f8d7da; color: #721c24'
            else:
                 styles[cancellation_idx] = 'background-color: #d4edda; color: #155724'
                 
        except KeyError:
            pass

        return styles


    # تطبيق التنسيق على الجدول كله باستخدام Styler
    styled_df = style_df.style.apply(
        highlight_performance,
        axis=1, # تطبيق التلوين صف بصف
    ).format({
        'معدل الالتزام (نسبة)': '{:.2f}%',
        'معدل الإلغاء (نسبة)': '{:.2f}%',
        'متوسط وقت التسليم (دقيقة)': '{:.2f}',
        'الإنتاجية (TPH)': '{:.2f}',
        'الطلبات المنجزة': '{:,.0f}',
        'إجمالي الساعات أونلاين': '{:.2f}',
    })
    
    return styled_df


def analyze_performance(pivot_df):
    """
    تطبيق منطق العمل لإنشاء توصيات بناءً على المقارنة بالمتوسط.
    يستخدم قيم النسبة (0-1) من الجدول المحوري الأصلي للحساب.
    """
    recommendations = {}

    analysis_df = pivot_df.copy()
    
    # أسماء الأعمدة المستخدمة في التحليل
    ontime_col = 'معدل الالتزام (نسبة)'
    cancellation_col = 'معدل الإلغاء (نسبة)'
    delivery_time_col = 'متوسط وقت التسليم (دقيقة)'
    tph_col = 'الإنتاجية (TPH)'
    delivered_tasks_col = 'الطلبات المنجزة'
    online_hours_col = 'إجمالي الساعات أونلاين'

    # حساب المتوسطات للمقارنة
    avg_ontime = analysis_df[ontime_col].mean()
    avg_delivery_time = analysis_df[delivery_time_col].mean()
    avg_cancellation = analysis_df[cancellation_col].mean()
    avg_tph = analysis_df[tph_col].mean()

    # استخدام القيمة الثابتة للحساسية
    LOW_PERFORMANCE_THRESHOLD = PERFORMANCE_THRESHOLD 
    HIGH_PERFORMANCE_THRESHOLD = 1 / PERFORMANCE_THRESHOLD 

    for index, row in analysis_df.iterrows():
        agent_name = row['اسم المندوب']
        notes = []

        # 1. تحليل كفاءة التسليم والالتزام بالوقت
        if row[ontime_col] < (avg_ontime * LOW_PERFORMANCE_THRESHOLD):
            notes.append(f"**🔴 انخفاض الالتزام بالوقت:** معدله {row[ontime_col]*100:.2f}% (أقل من متوسط الفريق). **التوصية:** تدريب على إدارة المسارات والبدء في الحركة بمجرد تأكيد الطلب.")
        
        # 2. تحليل سرعة التسليم
        if row[delivery_time_col] > (avg_delivery_time * HIGH_PERFORMANCE_THRESHOLD):
            notes.append(f"**🟡 ارتفاع متوسط وقت التسليم:** متوسطه {row[delivery_time_col]:.2f} دقيقة (أبطأ من المتوسط). **التوصية:** مراجعة سلوكه أثناء الاستلام والتسليم لتحديد نقاط الضعف.")

        # 3. تحليل معدل الإلغاء
        # إذا كان أعلى من المتوسط بـ 10% وأعلى من 2% (لتجنب التنبيه على قيم قليلة جداً)
        if row[cancellation_col] > (avg_cancellation * HIGH_PERFORMANCE_THRESHOLD) and row[cancellation_col] * 100 > 2:
            notes.append(f"**❌ معدل إلغاء مرتفع:** معدله {row[cancellation_col]*100:.2f}%. **التوصية:** التحقيق في سبب الإلغاءات المتكررة (أخطاء في الاستلام أو مشاكل في التواصل).")

        # 4. تحليل الإنتاجية (Tasks Per Hour)
        if row[tph_col] < (avg_tph * LOW_PERFORMANCE_THRESHOLD) and row[online_hours_col] > 5: # نراجع فقط من عمل أكثر من 5 ساعات
            notes.append(f"**📉 إنتاجية منخفضة (TPH):** يحقق {row[tph_col]:.2f} طلب/ساعة. **التوصية:** توجيهه للعمل في أوقات الذروة أو مراجعة منطق قبول الطلبات لديه.")

        # تجميع الملاحظات
        if notes:
            recommendations[agent_name] = {'ID': row['هوية المندوب (ID)'], 'Notes': notes}

    return recommendations

def to_excel(df):
    """دالة تحويل DataFrame إلى ملف Excel في الذاكرة لتمكين التصدير."""
    output = BytesIO()
    
    # تحويل النسب الداخلية (0-1) إلى نسب مئوية (0-100) مع الرمز % للتصدير
    export_df = df.copy()
    export_df['معدل الالتزام (%)'] = (export_df.pop('معدل الالتزام (نسبة)') * 100).round(2).astype(str) + '%'
    export_df['معدل الإلغاء (%)'] = (export_df.pop('معدل الإلغاء (نسبة)') * 100).round(2).astype(str) + '%'
    
    # إعادة ترتيب الأعمدة لتضمين التنسيق الجديد
    final_cols = ['هوية المندوب (ID)', 'اسم المندوب', 'الطلبات المنجزة', 'إجمالي الساعات أونلاين', 'الإنتاجية (TPH)',
                  'معدل الالتزام (%)', 'متوسط وقت التسليم (دقيقة)', 'معدل الإلغاء (%)']
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df[final_cols].to_excel(writer, index=False, sheet_name='Keeta_Delivery_Report_Summary')
            
    processed_data = output.getvalue()
    return processed_data

# ==============================================================================
# 3. واجهة التطبيق الرئيسية (Streamlit)
# ==============================================================================

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="أداة تحليل أداء مناديب كيتا")
st.title("🛵 محلل أداء مناديب التوصيل المتقدم (كيتا)")
st.markdown("---")
st.markdown("✅ **تم التحديث:** تم تجاهل المناديب الذين لم يسجلوا أي ساعة عمل (`Online Time = 0`) وتم إضافة تنسيق شرطي قوي.")

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
             df = pd.read_csv(uploaded_file)
        else:
             df = pd.read_excel(uploaded_file)
        
        # 1. تنظيف ومعالجة البيانات
        initial_count = len(df)
        df = clean_and_process_data(df)
        
        filtered_count = initial_count - len(df)
        st.success(f"تم تحميل الملف **{uploaded_file.name}** بنجاح. تم استبعاد **{filtered_count}** سجل (لعدم وجود ساعات عمل).")
        
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
        st.markdown("يرجى التأكد من أن الملف يحتوي على أعمدة الهوية والساعات أونلاين بالأسماء الصحيحة.")
    except Exception as e:
        st.error(f"❌ حدث خطأ غير متوقع أثناء المعالجة: {e}")
        st.markdown("**نصيحة:** قد يكون هناك مشكلة في تنسيق البيانات داخل الملف أو في الأعمدة المحفوظة.")
else:
    st.info("الرجاء رفع ملف الإكسيل أو CSV للبدء في تحليل أداء المناديب.")
