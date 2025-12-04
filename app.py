import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np

# ==============================================================================
# 1. الدوال المساعدة لتحميل ومعالجة البيانات
# ==============================================================================

def clean_and_process_data(uploaded_file):
    """
    تنظيف وتوحيد أسماء الأعمدة وتحويل البيانات للتحليل.
    يتم قراءة الملف هنا، مع محاولة تخطي الصفوف المدمجة في الهيدر.
    """
    
    # 1. محاولة قراءة الملف مع تخطي الصفوف الأولى (عادة لتفادي الـ Merged Cells)
    if uploaded_file.name.endswith('.csv'):
        # لملفات CSV، القراءة مباشرة أفضل
        df = pd.read_csv(uploaded_file, header=0)
    else:
        # لملفات Excel، جرب القراءة من الصف الأول (header=0) والصف الثاني (header=1)
        # إذا لم يكن هناك header=0، يتم محاولة قراءة header=1
        try:
            # محاولة القراءة بافتراض الهيدر في الصف الأول (الافتراضي)
            df = pd.read_excel(uploaded_file, header=0)
        except:
            # محاولة قراءة الملف مع اعتبار الصف الثاني هو الهيدر (لتخطي أي صفوف مدمجة)
            df = pd.read_excel(uploaded_file, header=1)
            
    # تنظيف أسماء الأعمدة من المسافات الزائدة والحروف الخاصة
    df.columns = df.columns.astype(str).str.strip().str.replace('[^a-zA-Z0-9\s%()]', '', regex=True).str.replace('\s+', ' ', regex=True)
    
    # تحديد الأعمدة الأساسية المطلوبة والأسماء القياسية الجديدة
    # نستخدم مجموعة أوسع من الأعمدة لتكون عملية التجميع في pivot_table أكثر شمولاً
    required_cols_map = {
        'Courier ID': 'ID',
        'Courier First Name': 'First Name',
        'Courier Last Name': 'Last Name',
        'Valid Online Time': 'Online Time (h)', 
        'Delivered Tasks': 'Delivered Tasks',
        'On-time Rate D': 'On-time Rate',
        'Avg Delivery Time of Delivered Orders': 'Avg Delivery Time (min)',
        'Cancellation Rate from Delivery Issues': 'Cancellation Rate'
    }
    
    # محاولة مطابقة الأعمدة المتاحة
    current_cols = {}
    for original, standard in required_cols_map.items():
        # البحث عن العمود بالاسم الأصلي أو اسم قريب
        if original in df.columns:
            current_cols[original] = standard
        else:
            # بحث مرن لأعمدة مثل On-time Rate (D)
            for col in df.columns:
                if original.split('(')[0].strip() in col:
                    current_cols[col] = standard
                    break
    
    df = df.rename(columns=current_cols, errors='ignore')

    # التأكد من تحويل الأعمدة الرقمية إلى النوع float
    for col in ['Online Time (h)', 'Delivered Tasks', 'On-time Rate', 'Avg Delivery Time (min)', 'Cancellation Rate']:
        if col in df.columns:
            # تنظيف أي رموز غير رقمية وتحويلها إلى أرقام
            df[col] = df[col].astype(str).str.replace('[^0-9.+-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # تصفية الصفوف التي لا تحتوي على ID للمندوب
    if 'ID' in df.columns:
        df = df.dropna(subset=['ID'])
        df['ID'] = df['ID'].astype(str)
    
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
    # تجنب القسمة على صفر
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
        'Avg_On_time_Rate': 'معدل الالتزام (%)',
        'Avg_Delivery_Time': 'متوسط وقت التسليم (دقيقة)',
        'Avg_Cancellation_Rate': 'معدل الإلغاء (%)'
    })
    
    # تنسيق النسب المئوية كأرقام لتطبيق الـ Highlighting (لتكون قيم بين 0 و 100)
    pivot_df['معدل الالتزام (%)'] = (pivot_df['معدل الالتزام (%)'] * 100).round(2)
    pivot_df['معدل الإلغاء (%)'] = (pivot_df['معدل الإلغاء (%)'] * 100).round(2)
    
    # ترتيب الأعمدة للعرض النهائي (الاستبعاد المؤقت للأسماء الأولى والأخيرة)
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
    
    # تعريف الحدود
    THRESHOLD = 0.8 # حد الأداء السيئ (20% أقل من المتوسط)
    
    def highlight_performance(s):
        """تطبيق التلوين على الأعمدة بناءً على المتوسط."""
        
        # معدل الالتزام (%) - قيمة أعلى = أفضل
        is_worst_ontime = s['معدل الالتزام (%)'] < avg_ontime * THRESHOLD
        
        # الإنتاجية (TPH) - قيمة أعلى = أفضل
        is_worst_tph = s['الإنتاجية (TPH)'] < avg_tph * THRESHOLD

        # متوسط وقت التسليم (دقيقة) - قيمة أقل = أفضل
        is_worst_delivery = s['متوسط وقت التسليم (دقيقة)'] > avg_delivery_time
        
        # معدل الإلغاء (%) - قيمة أقل = أفضل
        is_worst_cancellation = s['معدل الإلغاء (%)'] > avg_cancellation

        styles = [''] * len(s) # تهيئة قائمة التنسيقات بنفس حجم الصف
        
        # تحديد مؤشرات الأداء التي سيتم تطبيق التلوين عليها (مواقع الأعمدة)
        ontime_idx = df.columns.get_loc('معدل الالتزام (%)')
        tph_idx = df.columns.get_loc('الإنتاجية (TPH)')
        delivery_time_idx = df.columns.get_loc('متوسط وقت التسليم (دقيقة)')
        cancellation_idx = df.columns.get_loc('معدل الإلغاء (%)')
        
        # تطبيق التنسيق على العمود الصحيح
        # 1. معدل الالتزام (%)
        if is_worst_ontime:
            styles[ontime_idx] = 'background-color: #f8d7da; color: #721c24' # فاتح أحمر (سيئ)
        else:
            styles[ontime_idx] = 'background-color: #d4edda; color: #155724' # فاتح أخضر (جيد)

        # 2. الإنتاجية (TPH)
        if is_worst_tph:
            styles[tph_idx] = 'background-color: #f8d7da; color: #721c24' # فاتح أحمر (سيئ)
        else:
            styles[tph_idx] = 'background-color: #d4edda; color: #155724' # فاتح أخضر (جيد)

        # 3. وقت التسليم (دقيقة)
        if is_worst_delivery:
            styles[delivery_time_idx] = 'background-color: #f8d7da; color: #721c24' # فاتح أحمر (سيئ)
        else:
            styles[delivery_time_idx] = 'background-color: #d4edda; color: #155724' # فاتح أخضر (جيد)
            
        # 4. معدل الإلغاء (%)
        if is_worst_cancellation:
            styles[cancellation_idx] = 'background-color: #f8d7da; color: #721c24' # فاتح أحمر (سيئ)
        else:
            styles[cancellation_idx] = 'background-color: #d4edda; color: #155724' # فاتح أخضر (جيد)

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
    
    # استخدم الأعمدة الرقمية مباشرة من الجدول المحوري
    ontime_col = 'معدل الالتزام (%)'
    cancellation_col = 'معدل الإلغاء (%)'
    delivery_time_col = 'متوسط وقت التسليم (دقيقة)'
    tph_col = 'الإنتاجية (TPH)'
    delivered_tasks_col = 'الطلبات المنجزة'
    online_hours_col = 'إجمالي الساعات أونلاين'


    # حساب المتوسطات للمقارنة (بأخذ المتوسط من القيم غير المنسقة)
    avg_ontime = analysis_df[ontime_col].mean()
    avg_delivery_time = analysis_df[delivery_time_col].mean()
    avg_cancellation = analysis_df[cancellation_col].mean()
    avg_tph = analysis_df[tph_col].mean()

    # تعريف الحدود الدنيا/القصوى
    LOW_PERFORMANCE_THRESHOLD = 0.8  # 20% أقل من المتوسط
    HIGH_PERFORMANCE_THRESHOLD = 1.2 # 20% أعلى من المتوسط

    for index, row in analysis_df.iterrows():
        agent_name = row['اسم المندوب']
        notes = []

        # 1. تحليل كفاءة التسليم والالتزام بالوقت
        if row[ontime_col] < (avg_ontime * LOW_PERFORMANCE_THRESHOLD):
            notes.append(f"**🔴 انخفاض الالتزام بالوقت:** معدل التسليم في الموعد لديه هو {row[ontime_col]:.2f}% (أقل من المتوسط). **التوصية:** تدريب على إدارة المسارات والبدء في الحركة بمجرد تأكيد الطلب.")
        
        # 2. تحليل سرعة التسليم (ارتفاع الوقت سلبي)
        if row[delivery_time_col] > (avg_delivery_time * HIGH_PERFORMANCE_THRESHOLD) and row[delivered_tasks_col] > 0:
            notes.append(f"**🟡 ارتفاع متوسط وقت التسليم:** متوسطه هو {row[delivery_time_col]:.2f} دقيقة (أبطأ من المتوسط). **التوصية:** مراجعة سلوكه أثناء عملية الاستلام والتسليم لتحديد نقاط الضعف.")

        # 3. تحليل معدل الإلغاء (ارتفاع المعدل سلبي)
        # نضع حداً أدنى للإلغاء لا يزال يعتبر سيئاً حتى لو كان المتوسط منخفضاً جداً (مثلاً: فوق 5%)
        if row[cancellation_col] > (avg_cancellation * HIGH_PERFORMANCE_THRESHOLD) and row[cancellation_col] > 5:
            notes.append(f"**❌ معدل إلغاء مرتفع:** معدله هو {row[cancellation_col]:.2f}%. **التوصية:** التحقيق في سبب الإلغاءات (أخطاء متكررة في الاستلام أو مشاكل في التواصل).")

        # 4. تحليل الإنتاجية (Tasks Per Hour)
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
        export_df.to_excel(writer, index=False, sheet_name='Keeta_Delivery_Report', float_format='%.2f')
            
    processed_data = output.getvalue()
    return processed_data

# ==============================================================================
# 2. واجهة التطبيق الرئيسية (Streamlit)
# ==============================================================================

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="أداة تحليل أداء مناديب كيتا")
st.title("🛵 محلل أداء مناديب التوصيل المتقدم (كيتا)")
st.markdown("---")

# **التعديل الجديد:** استخدام st.file_uploader لتمكين التحميل المحلي.
uploaded_file = st.file_uploader("📥 **يرجى رفع ملف الإكسيل/CSV الخاص ببيانات المناديب**", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # 1. تنظيف ومعالجة البيانات
        df = clean_and_process_data(uploaded_file)
        
        st.success(f"تم تحميل الملف **{uploaded_file.name}** بنجاح. عدد السجلات: {len(df)}")
        
        # عرض البيانات الأولية
        st.subheader("📋 نموذج البيانات بعد المعالجة (أول 5 صفوف)")
        # التعديل السابق: إخفاء الـ index
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        st.markdown("---")

        # ==================================================
        # 2. إنشاء وعرض الجدول المحوري المنسق
        # ==================================================
        
        st.header("📈 تقرير أداء المناديب المجمّع (مُنسَّق)")
        pivot_table = generate_pivot_table(df)
        
        # تطبيق التنسيق الشرطي (Highlighting)
        styled_table = style_performance_table(pivot_table)
        
        # عرض الجدول المحوري المنسق (مع إخفاء الـ index)
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

    except KeyError as e:
        st.error(f"❌ خطأ في أسماء الأعمدة. يرجى التأكد من أن الأعمدة الأساسية (مثل Courier ID، Delivered Tasks، Valid Online Time) موجودة ومكتوبة بشكل صحيح في ملف الإكسيل الخاص بك.")
        st.error(f"العمود المفقود المحتمل: {e}")
    except Exception as e:
        st.error(f"❌ حدث خطأ غير متوقع أثناء المعالجة: {e}")
        st.markdown("**نصيحة:** تأكد أن الملف المرفوع هو ملف بيانات Excel/CSV صالح وبدون صفوف مدمجة كثيرة في البداية.")
else:
    st.info("الرجاء رفع ملف الإكسيل أو CSV للبدء في تحليل أداء المناديب.")
