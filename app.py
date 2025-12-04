import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np

# ==============================================================================
# 1. تعريف الهيدر الجديد وتحديد الثوابت
# ==============================================================================

# الهيدر الجديد الذي طلبه المستخدم (الأسماء القياسية باللغة الإنجليزية)
# هذه الأسماء هي التي سيتم استخدامها في الكود للتعامل مع البيانات
NEW_HEADER_NAMES = [
    'Date', 'Courier ID', 'Courier First Name', 'Courier Last Name', 'Supervisor', 
    'Vehicle Type', 'On-Shift?', 'Couriers Currently on Shift', 'Online Couriers', 
    'Courier App Online Time', 'Peak Online Hours', 'Accepted Tasks', 
    'Number of picked-up orders', 'Tasks with restaurant arrivals', 'Delivered Tasks', 
    'Large Order Tasks Completed', 'Cancelled Tasks', 'Rejected Tasks', 
    'Rejected Tasks (Courier)', 'Rejected Tasks (Auto)', 'On-time Rate (D)', 
    'Large order on-time rate', 'Avg Delivery Time of Delivered Orders', 
    'Delivered Orders Prop. (Over 55min)'
]

# *** هذا هو الجزء الذي يمكنك تعديله لتغيير حساسية التلوين والتوصيات ***
# ====================================================================
# القيمة التالية تحدد متى نعتبر الأداء سيئًا مقارنة بالمتوسط.
# 0.90 تعني أن الأداء سيئ إذا كان أقل من 90% من المتوسط (أي أقل بـ 10%)
PERFORMANCE_THRESHOLD = 0.90 
# ====================================================================

# ==============================================================================
# 2. الدوال المساعدة لتحميل ومعالجة البيانات
# ==============================================================================

def clean_and_process_data(uploaded_file):
    """
    تنظيف وتوحيد أسماء الأعمدة وتحويل البيانات للتحليل.
    تجاهل الصفوف العليا والاعتماد على الهيدر الجديد.
    """
    
    # تحديد عدد الصفوف المراد تجاهلها (عادة أول صفين يحتويان على عناوين مدمجة)
    skip_rows_count = 2

    # 1. محاولة قراءة الملف مع تخطي الصفوف الأولى وتعيين الهيدر يدوياً
    if uploaded_file.name.endswith('.csv'):
        # قراءة CSV مع تجاهل الصفوف
        # header=None يعني أننا نقول لـ Pandas لا يوجد هيدر، ثم نستخدم names لتحديد الهيدر الخاص بنا
        df = pd.read_csv(uploaded_file, skiprows=skip_rows_count, header=None, names=NEW_HEADER_NAMES)
    else:
        # قراءة Excel مع تجاهل الصفوف
        df = pd.read_excel(uploaded_file, skiprows=skip_rows_count, header=None, names=NEW_HEADER_NAMES)
            
    # تنظيف الأعمدة الرقمية
    numeric_cols = [
        'Courier App Online Time', 'Peak Online Hours', 'Accepted Tasks', 
        'Number of picked-up orders', 'Tasks with restaurant arrivals', 'Delivered Tasks', 
        'Large Order Tasks Completed', 'Cancelled Tasks', 'Rejected Tasks', 
        'Rejected Tasks (Courier)', 'Rejected Tasks (Auto)', 'On-time Rate (D)', 
        'Large order on-time rate', 'Avg Delivery Time of Delivered Orders', 
        'Delivered Orders Prop. (Over 55min)'
    ]

    for col in numeric_cols:
        if col in df.columns:
            # تنظيف أي رموز غير رقمية وتحويلها إلى أرقام
            # يجب الانتباه إلى أن بعض الأعمدة قد تكون بالفعل أرقام، لكن هذا يضمن التنظيف
            df[col] = df[col].astype(str).str.replace('[^0-9.+-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # تصفية الصفوف التي لا تحتوي على ID للمندوب
    if 'Courier ID' in df.columns:
        df = df.dropna(subset=['Courier ID'])
        df['Courier ID'] = df['Courier ID'].astype(str)
    
    return df

def generate_pivot_table(df):
    """ينشئ الجدول المحوري (Pivot Table) بتجميع مؤشرات الأداء."""
    
    # تجميع البيانات حسب المندوب
    pivot_df = df.groupby(['Courier ID', 'Courier First Name', 'Courier Last Name']).agg(
        Total_Delivered_Tasks=('Delivered Tasks', 'sum'),
        Total_Online_Hours=('Courier App Online Time', 'sum'),
        Avg_On_time_Rate=('On-time Rate (D)', 'mean'),
        Avg_Delivery_Time=('Avg Delivery Time of Delivered Orders', 'mean'),
        Total_Cancelled_Tasks=('Cancelled Tasks', 'sum') # حساب مجموع الإلغاءات
    ).reset_index()

    # إنشاء عمود الاسم الكامل
    pivot_df['Agent Name'] = pivot_df['Courier First Name'] + ' ' + pivot_df['Courier Last Name']
    
    # حساب الإنتاجية (Tasks Per Hour)
    pivot_df['Tasks Per Hour'] = np.where(
        pivot_df['Total_Online_Hours'] > 0,
        (pivot_df['Total_Delivered_Tasks'] / pivot_df['Total_Online_Hours']),
        0
    ).round(2)

    # حساب معدل الإلغاء (Cancellation Rate)
    # معدل الإلغاء = (الإلغاءات / (الإلغاءات + المنجزة)) * 100
    pivot_df['Cancellation Rate'] = np.where(
        (pivot_df['Total_Cancelled_Tasks'] + pivot_df['Total_Delivered_Tasks']) > 0,
        (pivot_df['Total_Cancelled_Tasks'] / (pivot_df['Total_Cancelled_Tasks'] + pivot_df['Total_Delivered_Tasks'])),
        0
    )
    
    # إعادة تسمية الأعمدة لتكون باللغة العربية
    pivot_df = pivot_df.rename(columns={
        'Courier ID': 'هوية المندوب (ID)',
        'Agent Name': 'اسم المندوب',
        'Total_Delivered_Tasks': 'الطلبات المنجزة',
        'Total_Online_Hours': 'إجمالي الساعات أونلاين',
        'Tasks Per Hour': 'الإنتاجية (TPH)',
        'Avg_On_time_Rate': 'معدل الالتزام (%)',
        'Avg_Delivery_Time': 'متوسط وقت التسليم (دقيقة)',
        'Cancellation Rate': 'معدل الإلغاء (%)' # هذا العمود تم حسابه
    })
    
    # تنسيق النسب المئوية كأرقام لتطبيق الـ Highlighting (لتكون قيم بين 0 و 100)
    pivot_df['معدل الالتزام (%)'] = (pivot_df['معدل الالتزام (%)'] * 100).round(2)
    pivot_df['معدل الإلغاء (%)'] = (pivot_df['معدل الإلغاء (%)'] * 100).round(2)
    
    # ترتيب الأعمدة للعرض النهائي
    final_cols = ['هوية المندوب (ID)', 'اسم المندوب', 'الطلبات المنجزة', 'إجمالي الساعات أونلاين', 'الإنتاجية (TPH)',
                  'معدل الالتزام (%)', 'متوسط وقت التسليم (دقيقة)', 'معدل الإلغاء (%)']
    
    pivot_df = pivot_df[[col for col in final_cols if col in pivot_df.columns]]
    
    return pivot_df

def style_performance_table(df):
    """
    تطبيق التنسيق الشرطي (Conditional Highlighting) على جدول الأداء.
    الأخضر = أداء جيد، الأحمر = أداء سيئ.
    """
    
    # حساب المتوسطات للمقارنة
    avg_ontime = df['معدل الالتزام (%)'].mean()
    avg_delivery_time = df['متوسط وقت التسليم (دقيقة)'].mean()
    avg_cancellation = df['معدل الإلغاء (%)'].mean()
    avg_tph = df['الإنتاجية (TPH)'].mean()
    
    # استخدام القيمة الثابتة للحساسية
    THRESHOLD = PERFORMANCE_THRESHOLD 
    
    def highlight_performance(s):
        """تطبيق التلوين على الأعمدة بناءً على المتوسط."""
        
        # 1. معدل الالتزام (%) - قيمة أعلى = أفضل (سيئ إذا كان أقل من المتوسط بـ 10%)
        is_worst_ontime = s['معدل الالتزام (%)'] < avg_ontime * THRESHOLD
        
        # 2. الإنتاجية (TPH) - قيمة أعلى = أفضل (سيئ إذا كان أقل من المتوسط بـ 10%)
        is_worst_tph = s['الإنتاجية (TPH)'] < avg_tph * THRESHOLD

        # 3. متوسط وقت التسليم (دقيقة) - قيمة أقل = أفضل (سيئ إذا كان أبطأ من المتوسط بأكثر من 10%)
        is_worst_delivery = s['متوسط وقت التسليم (دقيقة)'] > avg_delivery_time * (1 / THRESHOLD)
        
        # 4. معدل الإلغاء (%) - قيمة أقل = أفضل (سيئ إذا كان أعلى من المتوسط بأكثر من 10%)
        is_worst_cancellation = s['معدل الإلغاء (%)'] > avg_cancellation * (1 / THRESHOLD)

        styles = [''] * len(s) 
        
        # تحديد مؤشرات الأداء التي سيتم تطبيق التلوين عليها
        try:
            ontime_idx = df.columns.get_loc('معدل الالتزام (%)')
            tph_idx = df.columns.get_loc('الإنتاجية (TPH)')
            delivery_time_idx = df.columns.get_loc('متوسط وقت التسليم (دقيقة)')
            cancellation_idx = df.columns.get_loc('معدل الإلغاء (%)')
            
            # تطبيق التنسيق
            
            # 1. معدل الالتزام (%)
            styles[ontime_idx] = 'background-color: #f8d7da; color: #721c24' if is_worst_ontime else 'background-color: #d4edda; color: #155724'

            # 2. الإنتاجية (TPH)
            styles[tph_idx] = 'background-color: #f8d7da; color: #721c24' if is_worst_tph else 'background-color: #d4edda; color: #155724'

            # 3. وقت التسليم (دقيقة)
            styles[delivery_time_idx] = 'background-color: #f8d7da; color: #721c24' if is_worst_delivery else 'background-color: #d4edda; color: #155724'
                
            # 4. معدل الإلغاء (%)
            styles[cancellation_idx] = 'background-color: #f8d7da; color: #721c24' if is_worst_cancellation else 'background-color: #d4edda; color: #155724'
        except KeyError:
            # في حالة عدم وجود أي عمود، لن يتم تطبيق التنسيق
            pass

        return styles


    # تطبيق التنسيق على الجدول كله باستخدام Styler
    styled_df = df.style.apply(
        highlight_performance,
        axis=1, # تطبيق التلوين صف بصف
    ).format({
        'معدل الالتزام (%)': '{:.2f}%',
        'معدل الإلغاء (%)': '{:.2f}%',
        'متوسط وقت التسليم (دقيقة)': '{:.2f}',
        'الإنتاجية (TPH)': '{:.2f}',
        'الطلبات المنجزة': '{:,.0f}',
        'إجمالي الساعات أونلاين': '{:.2f}',
    })
    
    return styled_df


def analyze_performance(pivot_df):
    """تطبيق منطق العمل لإنشاء توصيات بناءً على المقارنة بالمتوسط."""
    recommendations = {}

    analysis_df = pivot_df.copy()
    
    # أسماء الأعمدة المستخدمة في التحليل
    ontime_col = 'معدل الالتزام (%)'
    cancellation_col = 'معدل الإلغاء (%)'
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
    HIGH_PERFORMANCE_THRESHOLD = 1 / PERFORMANCE_THRESHOLD # نستخدم مقلوب الـ Threshold للنقاط السلبية

    for index, row in analysis_df.iterrows():
        agent_name = row['اسم المندوب']
        notes = []

        # 1. تحليل كفاءة التسليم والالتزام بالوقت (أقل من المتوسط بـ 10% يعتبر سيئاً)
        if row[ontime_col] < (avg_ontime * LOW_PERFORMANCE_THRESHOLD):
            notes.append(f"**🔴 انخفاض الالتزام بالوقت:** معدل التسليم في الموعد لديه هو {row[ontime_col]:.2f}% (أقل من المتوسط). **التوصية:** تدريب على إدارة المسارات والبدء في الحركة بمجرد تأكيد الطلب.")
        
        # 2. تحليل سرعة التسليم (أكثر من المتوسط بـ 10% يعتبر سيئاً)
        if row[delivery_time_col] > (avg_delivery_time * HIGH_PERFORMANCE_THRESHOLD) and row[delivered_tasks_col] > 0:
            notes.append(f"**🟡 ارتفاع متوسط وقت التسليم:** متوسطه هو {row[delivery_time_col]:.2f} دقيقة (أبطأ من المتوسط). **التوصية:** مراجعة سلوكه أثناء عملية الاستلام والتسليم لتحديد نقاط الضعف.")

        # 3. تحليل معدل الإلغاء (أكثر من المتوسط بـ 10% يعتبر سيئاً)
        # نضع حداً أدنى للإلغاء لا يزال يعتبر سيئاً حتى لو كان المتوسط منخفضاً جداً (مثلاً: فوق 5%)
        if row[cancellation_col] > (avg_cancellation * HIGH_PERFORMANCE_THRESHOLD) and row[cancellation_col] > 5:
            notes.append(f"**❌ معدل إلغاء مرتفع:** معدله هو {row[cancellation_col]:.2f}%. **التوصية:** التحقيق في سبب الإلغاءات (أخطاء متكررة في الاستلام أو مشاكل في التواصل).")

        # 4. تحليل الإنتاجية (Tasks Per Hour) (أقل من المتوسط بـ 10% يعتبر سيئاً)
        if row[tph_col] < (avg_tph * LOW_PERFORMANCE_THRESHOLD) and row[online_hours_col] > 5:
            notes.append(f"**📉 إنتاجية منخفضة (TPH):** يحقق {row[tph_col]:.2f} طلب/ساعة. **التوصية:** توجيهه للعمل في أوقات الذروة لزيادة كفاءة ساعات عمله.")

        # تجميع الملاحظات
        if notes:
            recommendations[agent_name] = {'ID': row['هوية المندوب (ID)'], 'Notes': notes}

    return recommendations

def to_excel(df):
    """دالة تحويل DataFrame إلى ملف Excel في الذاكرة لتمكين التصدير."""
    output = BytesIO()
    
    # إنشاء نسخة قابلة للتصدير (إعادة التنسيق كنصوص)
    export_df = df.copy()
    
    # نحول النسب المئوية مرة أخرى إلى تنسيق نصي لإظهار الـ % في الإكسيل
    if 'معدل الالتزام (%)' in export_df.columns:
        export_df['معدل الالتزام (%)'] = export_df['معدل الالتزام (%)'].round(2).astype(str) + '%'
    if 'معدل الإلغاء (%)' in export_df.columns:
        export_df['معدل الإلغاء (%)'] = export_df['معدل الإلغاء (%)'].round(2).astype(str) + '%'

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # إزالة أي أعمدة غير ضرورية قبل التصدير
        cols_to_export = [col for col in df.columns if col not in ['Courier First Name', 'Courier Last Name']]
        export_df[cols_to_export].to_excel(writer, index=False, sheet_name='Keeta_Delivery_Report', float_format='%.2f')
            
    processed_data = output.getvalue()
    return processed_data

# ==============================================================================
# 3. واجهة التطبيق الرئيسية (Streamlit)
# ==============================================================================

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="أداة تحليل أداء مناديب كيتا")
st.title("🛵 محلل أداء مناديب التوصيل المتقدم (كيتا)")
st.markdown("---")
st.markdown("✅ **تم التحديث:** يستخدم المحلل الآن هيدر ثابت (كما طلبته) ويتجاهل الصفوف المدمجة لتجنب الأخطاء.")

# **التعديل الجديد:** استخدام st.file_uploader لتمكين التحميل المحلي.
uploaded_file = st.file_uploader("📥 **يرجى رفع ملف الإكسيل/CSV الخاص ببيانات المناديب**", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # 1. تنظيف ومعالجة البيانات
        df = clean_and_process_data(uploaded_file)
        
        st.success(f"تم تحميل الملف **{uploaded_file.name}** بنجاح. عدد السجلات: {len(df)}")
        
        # عرض البيانات الأولية
        st.subheader("📋 نموذج البيانات بعد المعالجة (أول 5 صفوف)")
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


        # زر تصدير الإكسيل
        st.download_button(
            label="⬇️ اضغط للتصدير كملف Excel",
            data=to_excel(pivot_table),
            file_name="Keeta_Delivery_Performance_Summary.xlsx",
            mime="application/vnd.ms-excel"
        )

        st.markdown("---")

        # ==================================================
        # 3. عرض التوصيات والتحليل
        # ==================================================
        
        st.header("📝 التوصيات ونوتات الأداء السيئ")
        recommendations = analyze_performance(pivot_table)

        if recommendations:
            st.warning("⚠️ **تنبيه:** تم تحديد المناديب التالية التي تحتاج إلى مراجعة أو تدريب:")
            
            # عرض التوصيات
            for agent, data in recommendations.items():
                st.markdown(f"### المندوب: {agent} (ID: {data['ID']})")
                for note in data['Notes']:
                    st.markdown(f"**- {note}**")
                st.markdown("---")
        else:
            st.balloons()
            st.success("🎉 **أداء ممتاز!** لا يوجد مناديب بأداء سيئ واضح خارج حدود التسامح المحددة.")

    except Exception as e:
        st.error(f"❌ حدث خطأ غير متوقع أثناء المعالجة: {e}")
        st.markdown("**نصيحة:** هذا الخطأ قد يعني أن هيكل ملف الإكسيل/CSV قد تغير بشكل كبير، أو أن الأعمدة لم تعد بالترتيب المتوقع بعد أول صفين. يرجى مراجعة الملف.")
else:
    st.info("الرجاء رفع ملف الإكسيل أو CSV للبدء في تحليل أداء المناديب.")
