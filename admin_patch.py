# ==============================================================================
# ADMIN PATCH  -  خطوات التركيب (بالترتيب):
#
#  1) في أول الملف غيّر:  layout="centered"  ->  layout="wide"
#  2) احذف الدالة القديمة auto_map() واستبدلها بـ "SECTION 1" تحت.
#  3) الصق "SECTION 2" (الدوال الجديدة + التحليلات + CSS الأدمن) قبل سطر:
#         # Admin page   (قبل def admin_upload_hunger)
#  4) احذف من def admin_upload_hunger() لحد آخر الملف
#     (يشمل admin_upload_keeta وبلوك if page != "Rider Performance")
#     والصق مكانهم "SECTION 3".
# ==============================================================================


# ==============================================================================
# SECTION 1  -  auto_map الجديدة (بتلقط عمود الـ id حتى لو اسمه مختلف)
# ==============================================================================
def auto_map(columns, fields=None):
    """يرجّع {field_key: اسم العمود في الملف أو None}"""
    fields = fields or FIELDS
    by_norm = {}
    for c in columns:
        by_norm.setdefault(norm(c), c)
    mapping = {}
    for key, (_, _, aliases) in fields.items():
        hit = next((by_norm[a] for a in aliases if a in by_norm), None)
        # للـ id: أي عمود اسمه بينتهي بـ id (Courier ID / Rider_ID / ID(Courier) ...)
        if hit is None and key == "rider_id":
            hit = next((c for n, c in by_norm.items()
                        if n.endswith("id") and len(n) <= 14), None)
        mapping[key] = hit
    return mapping


# ==============================================================================
# SECTION 2
# ==============================================================================

# ---------------- قراءة الملف (بتكتشف صف العناوين لوحدها) ----------------
def promote_header(grid, fields):
    """بعض الشيتات بيبقى فوقها عنوان/صفوف فاضية فالعناوين الحقيقية مش في أول صف."""
    grid = grid.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    alias = {a for _, (_, _, al) in fields.items() for a in al}
    best_i, best = 0, -1
    for i in range(min(15, len(grid))):
        s = sum(norm(v) in alias for v in grid.iloc[i] if pd.notna(v))
        if s > best:
            best, best_i = s, i
    hdr = [str(v).replace("\ufeff", "").strip() if pd.notna(v) else f"col_{j}"
           for j, v in enumerate(grid.iloc[best_i])]
    seen, uniq = {}, []
    for h in hdr:
        seen[h] = seen.get(h, 0) + 1
        uniq.append(h if seen[h] == 1 else f"{h}.{seen[h] - 1}")
    raw = grid.iloc[best_i + 1:].copy()
    raw.columns = uniq
    return raw.reset_index(drop=True), best_i


def load_file_ui(prefix, fields):
    """يرفع الملف ويرجّع (raw, mapping) أو None. كل القيم نصوص عشان الـ ID الطويل ميتبوظش."""
    up = st.file_uploader("Excel / CSV", type=["xlsx", "xls", "csv"], key=f"up_{prefix}")
    if not up:
        return None
    try:
        if up.name.lower().endswith(".csv"):
            try:
                grid = pd.read_csv(up, header=None, dtype=str, encoding="utf-8-sig")
            except Exception:
                up.seek(0)
                grid = pd.read_csv(up, header=None, dtype=str, encoding="latin1")
            sheet = "csv"
        else:
            xl = pd.ExcelFile(up)
            names = xl.sheet_names
            sheet = names[0]
            if len(names) > 1:
                sheet = st.selectbox("الشيت", names, key=f"sheet_{prefix}_{up.name}")
            grid = xl.parse(sheet, header=None, dtype=str)
    except Exception as e:
        st.error(f"تعذّر قراءة الملف: {e}")
        return None

    raw, hrow = promote_header(grid, fields)
    if raw.empty:
        st.error("الشيت فاضي.")
        return None
    note = f" — تم اكتشاف العناوين في الصف رقم {hrow + 1}" if hrow > 0 else ""
    st.markdown(
        f'<div class="adm-chip ok">✔ {up.name} · {len(raw):,} صف · {len(raw.columns)} عمود{note}</div>',
        unsafe_allow_html=True)

    mapping = auto_map(raw.columns, fields)
    mapping = mapping_editor(raw.columns, fields, mapping, f"{prefix}_{up.name}_{sheet}")
    return raw, mapping


def mapping_editor(columns, fields, mapping, prefix):
    """يعرض مطابقة الأعمدة ويسمح للأدمن يختار يدوي لو الاسم مختلف."""
    opts = ["— غير موجود —"] + list(columns)
    need_attention = mapping.get("rider_id") is None
    with st.expander("🧭 مطابقة الأعمدة" + ("  ⚠️ اختر عمود الـ ID" if need_attention else ""),
                     expanded=need_attention):
        cols = st.columns(3)
        for i, (key, (label, required, _)) in enumerate(fields.items()):
            cur = mapping.get(key)
            idx = opts.index(cur) if cur in opts else 0
            pick = cols[i % 3].selectbox(label + (" *" if required else ""), opts, index=idx,
                                         key=f"map_{prefix}_{key}")
            mapping[key] = None if pick == opts[0] else pick
    return mapping


# ---------------- مقارنة بالرفع السابق ----------------
def compare_with_previous(df, project):
    old = get_all(project)
    if old.empty:
        st.markdown('<div class="adm-chip">أول رفع للمشروع ده — مفيش بيانات سابقة للمقارنة.</div>',
                    unsafe_allow_html=True)
        return
    new_ids, old_ids = set(df["rider_id"]), set(old["rider_id"])
    st.markdown(
        '<div class="adm-kpis">'
        + kpi_html("رايدرز جداد", len(new_ids - old_ids), "مش موجودين في الرفع السابق", "#16a34a")
        + kpi_html("رايدرز اختفوا", len(old_ids - new_ids), "كانوا في الرفع السابق", "#dc2626")
        + kpi_html("مستمرين", len(new_ids & old_ids), "موجودين في الاتنين", "#2563eb")
        + "</div>", unsafe_allow_html=True)


# ---------------- مكونات UI ----------------
def kpi_html(label, value, sub="", color="#2563eb"):
    return (f'<div class="adm-kpi" style="--c:{color}"><div class="adm-kpi-l">{html.escape(str(label))}</div>'
            f'<div class="adm-kpi-v">{value}</div><div class="adm-kpi-s">{html.escape(str(sub))}</div></div>')


def kpis(items):
    st.markdown('<div class="adm-kpis">' + "".join(kpi_html(*i) for i in items) + "</div>",
                unsafe_allow_html=True)


def sec(title):
    st.markdown(f'<div class="adm-sec">{html.escape(title)}</div>', unsafe_allow_html=True)


def bar(series, height=260):
    if series is None or len(series) == 0:
        st.caption("لا توجد بيانات.")
        return
    s = series.copy()
    s.index = s.index.astype(str)
    st.bar_chart(s, height=height)


def pct_buckets(s, edges, labels):
    return pd.cut(s, edges, labels=labels, include_lowest=True).value_counts().reindex(labels).fillna(0).astype(int)


def inject_admin_css():
    st.markdown("""
<style>
    .block-container {max-width: 1280px !important;}
    .adm-hero {direction:rtl; text-align:right; background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 100%);
               color:#fff; border-radius:24px; padding:26px 30px; margin:6px 0 18px;
               display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
               box-shadow:0 14px 40px rgba(15,23,42,.22);}
    .adm-hero h1 {margin:0; font-size:28px; font-weight:800; color:#fff;}
    .adm-hero p {margin:4px 0 0; color:#cbd5e1; font-size:14px;}
    .adm-pill {background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25); border-radius:999px;
               padding:8px 16px; font-weight:700; font-size:14px;}
    .adm-kpis {direction:rtl; display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
               gap:12px; margin:8px 0 18px;}
    .adm-kpi {background:#fff; border:1px solid #e8ecf4; border-radius:18px; padding:14px 18px;
              border-top:5px solid var(--c); box-shadow:0 6px 18px rgba(15,23,42,.05); text-align:right;}
    .adm-kpi-l {color:#64748b; font-weight:700; font-size:13px;}
    .adm-kpi-v {font-size:30px; font-weight:800; color:#0f172a; line-height:1.2;}
    .adm-kpi-s {color:#94a3b8; font-size:12px; font-weight:600;}
    .adm-sec {direction:rtl; text-align:right; font-size:18px; font-weight:800; color:#0f172a;
              margin:26px 0 10px; padding-right:12px; border-right:5px solid #2563eb;}
    .adm-chip {direction:rtl; text-align:right; display:inline-block; background:#f1f5f9; color:#334155;
               border-radius:12px; padding:8px 14px; font-size:13px; font-weight:600; margin:6px 0;}
    .adm-chip.ok {background:#f0fdf4; color:#166534; border:1px solid #bbf7d0;}
    .adm-chip.warn {background:#fffbeb; color:#92400e; border:1px solid #fde68a;}
    .adm-steps {direction:rtl; display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 14px;}
    .adm-step {background:#f1f5f9; color:#475569; border-radius:999px; padding:6px 14px; font-size:13px; font-weight:700;}
    .adm-step b {background:#0f172a; color:#fff; border-radius:50%; display:inline-block; width:20px; height:20px;
                 text-align:center; line-height:20px; margin-left:6px; font-size:11px;}
    .adm-proj {direction:rtl; text-align:right; background:#fff; border:1px solid #e8ecf4; border-radius:20px;
               padding:18px 20px; box-shadow:0 6px 18px rgba(15,23,42,.05); margin-bottom:12px;}
    .adm-proj h3 {margin:0 0 4px; font-size:20px;}
    .adm-proj small {color:#94a3b8;}
</style>
""", unsafe_allow_html=True)


def admin_header(pending):
    st.markdown(f"""
<div class="adm-hero">
  <div><h1>⚙️ لوحة التحكم</h1><p>رفع التقارير · تحليل الأداء · إدارة الطلبات</p></div>
  <div class="adm-pill">📨 طلبات قيد المراجعة: {pending}</div>
</div>""", unsafe_allow_html=True)


def steps_bar(active):
    names = ["رفع الملف", "مطابقة الأعمدة", "التحليلات", "الحفظ"]
    h = "".join(
        f'<div class="adm-step" style="{"background:#2563eb;color:#fff" if i + 1 == active else ""}">'
        f'<b>{i + 1}</b>{n}</div>' for i, n in enumerate(names))
    st.markdown(f'<div class="adm-steps">{h}</div>', unsafe_allow_html=True)


# ---------------- تحليلات هنجر ستيشن ----------------
def render_hunger_analytics(df):
    if df.empty:
        st.info("لا توجد بيانات.")
        return
    n = len(df)
    active = df[df["orders"] > 0]
    ot_ok = int((df["on_time_rate"] >= 98).sum())
    ver_ok = int((df["verification_rate"] >= 100).sum())
    fail_ok = int((df["fail_rate"] >= 100).sum())

    sec("📌 الملخص العام")
    kpis([
        ("عدد الرايدرز", f"{n:,}", f"نشط (له طلبات): {len(active):,}", "#2563eb"),
        ("إجمالي الطلبات", f"{int(df['orders'].sum()):,}", f"متوسط/رايدر: {df['orders'].mean():.1f}", "#7c3f2b"),
        ("متوسط On-Time", fmt_pct(df["on_time_rate"].mean()), "المطلوب 98%", "#16a34a"),
        ("متوسط التحقق", fmt_pct(df["verification_rate"].mean()), "المطلوب 100%", "#8b5cf6"),
        ("متوسط Fail Order", fmt_pct(df["fail_rate"].mean()), "المطلوب 100%", "#f59e0b"),
        ("متوسط الجودة النهائية", fmt_pct(df["final_score"].mean()), "", "#0ea5e9"),
    ])

    sec("🎯 الالتزام بالمستهدف")
    kpis([
        ("حققوا On-Time", f"{ot_ok:,}", f"{ot_ok / n:.0%} من الرايدرز", "#16a34a"),
        ("حققوا التحقق", f"{ver_ok:,}", f"{ver_ok / n:.0%} من الرايدرز", "#16a34a"),
        ("حققوا Fail Order", f"{fail_ok:,}", f"{fail_ok / n:.0%} من الرايدرز", "#16a34a"),
        ("تحت On-Time", f"{n - ot_ok:,}", "يحتاجوا متابعة", "#dc2626"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 توزيع السيجمنت")
        bar(df["segment"].value_counts().sort_index())
    with c2:
        sec("⏱️ توزيع On-Time")
        bar(pct_buckets(df["on_time_rate"], [-0.01, 80, 90, 95, 98, 100.01],
                        ["<80", "80-90", "90-95", "95-98", "98+"]))

    if "target_per_day" in df.columns and "days_covered" in df.columns:
        t, d = int(df["target_per_day"].iloc[0]), int(df["days_covered"].iloc[0])
        exp = t * d
        hit = int((df["orders"] >= exp).sum())
        sec("🎯 التارجت اليومي")
        kpis([
            ("المطلوب لكل رايدر", f"{exp:,}", f"{t} × {d} يوم", "#2563eb"),
            ("محققين التارجت", f"{hit:,}", f"{hit / n:.0%}", "#16a34a"),
            ("تحت التارجت", f"{n - hit:,}", f"{(n - hit) / n:.0%}", "#dc2626"),
            ("إجمالي النقص", f"{int((exp - df['orders']).clip(lower=0).sum()):,}", "طلب", "#f59e0b"),
        ])

    if "prev_segment" in df.columns:
        sec("🔄 حركة السيجمنت مقارنة بالرفع السابق")
        order = {s: i for i, s in enumerate("ABCDEF")}
        cur = df["segment"].astype(str).str.upper().map(order)
        prv = df["prev_segment"].astype(str).str.upper().map(order)
        both = cur.notna() & prv.notna()
        up, down = int((both & (cur < prv)).sum()), int((both & (cur > prv)).sum())
        same = int((both & (cur == prv)).sum())
        kpis([("⬆️ ارتفعوا", up, "", "#16a34a"), ("⬇️ انخفضوا", down, "", "#dc2626"),
              ("➡️ ثابتين", same, "", "#2563eb"), ("بدون سابق", int((~both).sum()), "", "#94a3b8")])

    cols = ["rider_id", "rider_name", "orders", "on_time_rate", "verification_rate", "fail_rate", "segment"]
    c3, c4 = st.columns(2)
    with c3:
        sec("🏅 أفضل 10 (On-Time)")
        st.dataframe(active.nlargest(10, "on_time_rate")[cols], hide_index=True, use_container_width=True)
    with c4:
        sec("⚠️ أضعف 10 (On-Time)")
        st.dataframe(active.nsmallest(10, "on_time_rate")[cols], hide_index=True, use_container_width=True)

    sec("🧹 جودة البيانات")
    kpis([
        ("بدون طلبات", int((df["orders"] == 0).sum()), "", "#f59e0b"),
        ("بدون اسم", int((df["rider_name"].astype(str).str.strip() == "").sum()), "", "#f59e0b"),
        ("بدون سيجمنت", int((df["segment"].astype(str).isin(["—", "", "nan"])).sum()), "", "#f59e0b"),
    ])


# ---------------- تحليلات كيتا ----------------
def render_keeta_analytics(df):
    if df.empty:
        st.info("لا توجد بيانات.")
        return
    n = len(df)
    deliv, acc = int(df["delivered"].sum()), int(df["accepted"].sum())
    lvl = df["level"].astype(str).replace({"": "غير محدد", "nan": "غير محدد"})

    sec("📌 الملخص العام")
    kpis([
        ("عدد الرايدرز", f"{n:,}", "", "#2563eb"),
        ("المهام الموصلة", f"{deliv:,}", f"متوسط/رايدر: {df['delivered'].mean():.1f}", "#eab308"),
        ("المهام المقبولة", f"{acc:,}", f"نسبة الإنجاز: {(deliv / acc if acc else 0):.0%}", "#0ea5e9"),
        ("متوسط On-Time", fmt_pct(df["on_time_rate"].mean()), "", "#16a34a"),
        ("متوسط نسبة الإلغاء", fmt_pct(df["cancel_rate"].mean()), "", "#dc2626"),
        ("إجمالي المتأخر", f"{int(df['orders_late'].sum()):,}", "طلب", "#f59e0b"),
    ])

    sec("🚨 تنبيهات")
    kpis([
        ("إلغاء > 5%", int((df["cancel_rate"] > 5).sum()), "رايدر", "#dc2626"),
        ("On-Time < 90%", int(((df["on_time_rate"] < 90) & (df["delivered"] > 0)).sum()), "رايدر", "#f59e0b"),
        ("لديهم مرفوضات", int((df["rejected"] > 0).sum()), f"إجمالي {int(df['rejected'].sum()):,}", "#8b5cf6"),
        ("بدون مهام موصلة", int((df["delivered"] == 0).sum()), "", "#94a3b8"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 توزيع المستويات (Level)")
        bar(lvl.value_counts().sort_index())
    with c2:
        sec("⏱️ توزيع On-Time")
        bar(pct_buckets(df["on_time_rate"], [-0.01, 70, 80, 90, 95, 100.01],
                        ["<70", "70-80", "80-90", "90-95", "95+"]))

    c3, c4 = st.columns(2)
    with c3:
        sec("📦 توزيع المهام الموصلة")
        bar(pct_buckets(df["delivered"], [-1, 0, 50, 100, 200, 400, 10**9],
                        ["0", "1-50", "51-100", "101-200", "201-400", "400+"]))
    with c4:
        sec("❌ توزيع نسبة الإلغاء")
        bar(pct_buckets(df["cancel_rate"], [-0.01, 0, 2, 5, 10, 100.01],
                        ["0%", "0-2%", "2-5%", "5-10%", "10%+"]))

    cols = ["rider_id", "rider_name", "delivered", "on_time_rate", "cancel_rate", "orders_late", "level"]
    act = df[df["delivered"] > 0]
    c5, c6 = st.columns(2)
    with c5:
        sec("🏅 أفضل 10 (المهام الموصلة)")
        st.dataframe(act.nlargest(10, "delivered")[cols], hide_index=True, use_container_width=True)
    with c6:
        sec("⚠️ أعلى 10 نسبة إلغاء")
        st.dataframe(act.nlargest(10, "cancel_rate")[cols], hide_index=True, use_container_width=True)

    sec("🧹 جودة البيانات")
    kpis([
        ("بدون valid day", int((df["valid_day"] == "").sum()), "", "#f59e0b"),
        ("بدون level", int((df["level"] == "").sum()), "", "#f59e0b"),
        ("بدون اسم", int((df["rider_name"].astype(str).str.strip() == "").sum()), "", "#f59e0b"),
    ])


# ---------------- رفع هنجر ستيشن ----------------
def admin_upload_hunger():
    st.markdown('<div class="adm-chip warn">⚠️ الرفع الجديد بيمسح بيانات هنجر ستيشن القديمة فقط (كيتا مش بتتأثر).</div>',
                unsafe_allow_html=True)
    steps_bar(1)
    res = load_file_ui("hunger", FIELDS)
    if not res:
        return
    raw, mapping = res
    if mapping["rider_id"] is None:
        st.error("❌ لازم تختار عمود الـ ID من «مطابقة الأعمدة» فوق.")
        return

    missing = [FIELDS[k][0] for k, v in mapping.items() if v is None]
    if missing:
        st.markdown('<div class="adm-chip warn">أعمدة غير موجودة (هتظهر 0 / —): ' + "، ".join(missing) + "</div>",
                    unsafe_allow_html=True)

    clean = build_clean_df(raw, mapping)
    if clean.empty:
        st.error("مفيش صفوف صالحة بعد التنظيف. اتأكد إن عمود الـ ID صح.")
        return

    sec("🎯 إعدادات التارجت")
    g1, g2 = st.columns(2)
    tgt = g1.number_input("التارجت اليومي (طلب/يوم)", min_value=1, value=17, step=1)
    days = g2.number_input("عدد الأيام اللي الشيت مغطيها", min_value=1, max_value=31,
                           value=max(datetime.now().day - 1, 1), step=1)
    clean["target_per_day"], clean["days_covered"] = int(tgt), int(days)

    sec("🔁 المقارنة بالرفع السابق")
    compare_with_previous(clean, "hunger")

    # السيجمنت السابق عشان تحليل الحركة يظهر قبل الحفظ
    if table_exists(TABLE):
        try:
            old = pd.read_sql_query(f"SELECT rider_id, segment AS prev_segment FROM {TABLE}", get_conn()
                                    ).drop_duplicates("rider_id")
            preview = clean.merge(old, on="rider_id", how="left")
        except Exception:
            preview = clean
    else:
        preview = clean

    render_hunger_analytics(preview)

    sec("💾 الحفظ")
    st.dataframe(clean.drop(columns=["uploaded_at"]).head(20), use_container_width=True, hide_index=True)
    if st.button("💾 مسح القديم ورفع الجديد", type="primary", use_container_width=True, key="save_hunger"):
        with_prev = replace_all_data(clean)
        st.success(f"✅ تم. اتحمّل {len(clean):,} رايدر ({with_prev:,} منهم ليهم سيجمنت سابق).")


# ---------------- رفع كيتا ----------------
def admin_upload_keeta():
    st.markdown('<div class="adm-chip warn">⚠️ الرفع الجديد بيمسح بيانات كيتا القديمة فقط (هنجر ستيشن مش بتتأثر).</div>',
                unsafe_allow_html=True)
    st.caption("التيمبلت: id | name | accepted | Delivered Tasks | cancel | Rejected Tasks | "
               "Cancellation Rate from Delivery Issues | Experience_On-time Rate (D) | orders late | valid day | level")
    steps_bar(1)
    res = load_file_ui("keeta", KEETA_FIELDS)
    if not res:
        return
    raw, mapping = res
    if mapping["rider_id"] is None:
        st.error("❌ لازم تختار عمود الـ ID من «مطابقة الأعمدة» فوق.")
        return

    missing = [KEETA_FIELDS[k][0] for k, v in mapping.items() if v is None]
    if missing:
        st.markdown('<div class="adm-chip warn">أعمدة غير موجودة (هتظهر 0 / —): ' + "، ".join(missing) + "</div>",
                    unsafe_allow_html=True)

    clean = build_keeta_df(raw, mapping)
    if clean.empty:
        st.error("مفيش صفوف صالحة بعد التنظيف. اتأكد إن عمود الـ ID صح.")
        return

    sec("🔁 المقارنة بالرفع السابق")
    compare_with_previous(clean, "keeta")
    render_keeta_analytics(clean)

    sec("💾 الحفظ")
    st.dataframe(clean.drop(columns=["uploaded_at"]).head(20), use_container_width=True, hide_index=True)
    if st.button("💾 مسح القديم ورفع الجديد", type="primary", use_container_width=True, key="save_keeta"):
        replace_keeta_data(clean)
        st.success(f"✅ تم. اتحمّل {len(clean):,} رايدر.")


# ---------------- لوحة النظرة العامة ----------------
def admin_dashboard():
    sec("🏠 نظرة عامة على المشاريع")
    c1, c2 = st.columns(2)
    for col, key, title, color in ((c1, "hunger", "🍔 HungerStation", "#7c3f2b"),
                                   (c2, "keeta", "🟡 Keeta", "#eab308")):
        df = get_all(key)
        pend = len([r for r in list_requests(status="pending", project=key, limit=1000)])
        with col:
            if df.empty:
                st.markdown(f'<div class="adm-proj"><h3>{title}</h3><small>لم يتم رفع بيانات بعد</small></div>',
                            unsafe_allow_html=True)
                continue
            st.markdown(f'<div class="adm-proj"><h3>{title}</h3><small>آخر رفع: {df["uploaded_at"].iloc[0]}</small></div>',
                        unsafe_allow_html=True)
            if key == "hunger":
                kpis([("الرايدرز", f"{len(df):,}", "", color),
                      ("متوسط On-Time", fmt_pct(df["on_time_rate"].mean()), "", "#16a34a"),
                      ("طلبات إجازة", pend, "قيد المراجعة", "#f59e0b")])
            else:
                kpis([("الرايدرز", f"{len(df):,}", "", color),
                      ("متوسط On-Time", fmt_pct(df["on_time_rate"].mean()), "", "#16a34a"),
                      ("طلبات إجازة", pend, "قيد المراجعة", "#f59e0b")])
    st.caption("افتح تاب «رفع تقرير» لرفع شيت جديد، أو «البيانات الحالية» لعرض التحليلات الكاملة.")


# ---------------- البيانات الحالية ----------------
def admin_data_tab():
    dp = st.radio("المشروع", ["Keeta", "HungerStation"], horizontal=True, key="data_proj")
    dp_key = "keeta" if dp == "Keeta" else "hunger"
    current = get_all(dp_key)
    if current.empty:
        st.info("لم يتم رفع بيانات لهذا المشروع بعد.")
        return
    st.markdown(f'<div class="adm-chip ok">آخر رفع: {current["uploaded_at"].iloc[0]} · {len(current):,} رايدر</div>',
                unsafe_allow_html=True)
    t_an, t_tb = st.tabs(["📈 التحليلات", "📋 جدول الرايدرز"])
    with t_an:
        (render_hunger_analytics if dp_key == "hunger" else render_keeta_analytics)(current)
    with t_tb:
        q = st.text_input("🔎 بحث بالـ ID أو الاسم", key="data_q").strip().lower()
        view = current.drop(columns=["uploaded_at"])
        if q:
            view = view[view["rider_id"].astype(str).str.lower().str.contains(q, regex=False)
                        | view["rider_name"].astype(str).str.lower().str.contains(q, regex=False)]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("📥 تحميل البيانات CSV", view.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{dp_key}_performance_current.csv", mime="text/csv",
                           use_container_width=True)


# ==============================================================================
# SECTION 3  -  بلوك الأدمن (آخر الملف)
# ==============================================================================
if page != "Rider Performance":
    if not admin_login():
        st.stop()

    inject_admin_css()
    _pend = count_requests("pending")
    admin_header(_pend)

    t_dash, t_up, t_data, t_req = st.tabs([
        "🏠 نظرة عامة", "📥 رفع تقرير", "📊 البيانات والتحليلات",
        "📨 الطلبات" + (f" ({_pend})" if _pend else "")])

    with t_dash:
        admin_dashboard()
    with t_up:
        up_proj = st.radio("المشروع", ["Keeta", "HungerStation"], horizontal=True, key="up_proj")
        (admin_upload_keeta if up_proj == "Keeta" else admin_upload_hunger)()
    with t_data:
        admin_data_tab()
    with t_req:
        admin_requests_ui()
