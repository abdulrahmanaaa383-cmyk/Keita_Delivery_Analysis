import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import hashlib
import html

# ============================================================
# RIDER PERFORMANCE PORTAL
# Public rider performance lookup + Admin Excel upload
# ============================================================

st.set_page_config(
    page_title="Rider Performance",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "rider_performance.db"

PERFORMANCE_COLUMNS = [
    "Month",
    "city_name",
    "contract_name",
    "rider_id",
    "vehicle_type",
    "total_verification_requests",
    "successful_verification_requests",
    "verification_success_rate",
    "gross_orders",
    "completed_orders",
    "completed_orders_in_time",
    "failed_orders_by_rider",
    "on_time_delivery_score",
    "fail_rate_score",
    "final_delivery_quality_score",
    "segment",
]

DISPLAY_NAMES = {
    "Month": "Month",
    "city_name": "City",
    "contract_name": "Contract",
    "rider_id": "Rider ID",
    "vehicle_type": "Vehicle",
    "total_verification_requests": "Verification Requests",
    "successful_verification_requests": "Successful Verification",
    "verification_success_rate": "Verification Score",
    "gross_orders": "Gross Orders",
    "completed_orders": "Completed Orders",
    "completed_orders_in_time": "Completed Orders In-Time",
    "late_orders": "Late Orders",
    "failed_orders_by_rider": "Failed Orders",
    "on_time_delivery_score": "On-Time Delivery Score",
    "fail_rate_score": "Fail Rate Score",
    "final_delivery_quality_score": "Final Delivery Quality Score",
    "segment": "Segment",
}

# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS riders (
            rider_id TEXT PRIMARY KEY,
            rider_name TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            rider_id TEXT NOT NULL,
            month TEXT,
            city_name TEXT,
            contract_name TEXT,
            vehicle_type TEXT,
            total_verification_requests INTEGER,
            successful_verification_requests INTEGER,
            verification_success_rate REAL,
            gross_orders INTEGER,
            completed_orders INTEGER,
            completed_orders_in_time INTEGER,
            failed_orders_by_rider INTEGER,
            on_time_delivery_score REAL,
            fail_rate_score REAL,
            final_delivery_quality_score REAL,
            segment TEXT,
            uploaded_at TEXT NOT NULL,
            PRIMARY KEY (rider_id, month, contract_name)
        )
    """)
    conn.commit()
    conn.close()


def save_names(df):
    if df.empty:
        return 0

    conn = get_conn()
    now = datetime.now().isoformat(timespec="seconds")
    count = 0

    for _, row in df.iterrows():
        rider_id = str(row["rider_id"]).strip()
        rider_name = str(row["rider_name"]).strip()

        if not rider_id or not rider_name or rider_name.lower() == "nan":
            continue

        conn.execute("""
            INSERT INTO riders (rider_id, rider_name, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(rider_id) DO UPDATE SET
                rider_name = excluded.rider_name,
                updated_at = excluded.updated_at
        """, (rider_id, rider_name, now))
        count += 1

    conn.commit()
    conn.close()
    return count


def load_names():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT rider_id, rider_name FROM riders ORDER BY rider_name",
        conn
    )
    conn.close()
    return df


def save_performance(df):
    conn = get_conn()
    now = datetime.now().isoformat(timespec="seconds")

    for _, row in df.iterrows():
        vals = [
            str(row.get("rider_id", "")).strip(),
            str(row.get("Month", "")),
            str(row.get("city_name", "")),
            str(row.get("contract_name", "")),
            str(row.get("vehicle_type", "")),
            to_int(row.get("total_verification_requests")),
            to_int(row.get("successful_verification_requests")),
            to_float(row.get("verification_success_rate")),
            to_int(row.get("gross_orders")),
            to_int(row.get("completed_orders")),
            to_int(row.get("completed_orders_in_time")),
            to_int(row.get("failed_orders_by_rider")),
            to_float(row.get("on_time_delivery_score")),
            to_float(row.get("fail_rate_score")),
            to_float(row.get("final_delivery_quality_score")),
            str(row.get("segment", "")),
            now,
        ]

        conn.execute("""
            INSERT INTO performance (
                rider_id, month, city_name, contract_name, vehicle_type,
                total_verification_requests, successful_verification_requests,
                verification_success_rate, gross_orders, completed_orders,
                completed_orders_in_time, failed_orders_by_rider,
                on_time_delivery_score, fail_rate_score,
                final_delivery_quality_score, segment, uploaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rider_id, month, contract_name) DO UPDATE SET
                city_name = excluded.city_name,
                vehicle_type = excluded.vehicle_type,
                total_verification_requests = excluded.total_verification_requests,
                successful_verification_requests = excluded.successful_verification_requests,
                verification_success_rate = excluded.verification_success_rate,
                gross_orders = excluded.gross_orders,
                completed_orders = excluded.completed_orders,
                completed_orders_in_time = excluded.completed_orders_in_time,
                failed_orders_by_rider = excluded.failed_orders_by_rider,
                on_time_delivery_score = excluded.on_time_delivery_score,
                fail_rate_score = excluded.fail_rate_score,
                final_delivery_quality_score = excluded.final_delivery_quality_score,
                segment = excluded.segment,
                uploaded_at = excluded.uploaded_at
        """, vals)

    conn.commit()
    conn.close()


def get_performance(rider_id):
    conn = get_conn()

    query = """
        SELECT
            p.*,
            COALESCE(r.rider_name, '') AS rider_name
        FROM performance p
        LEFT JOIN riders r ON p.rider_id = r.rider_id
        WHERE p.rider_id = ?
        ORDER BY p.uploaded_at DESC
        LIMIT 1
    """

    row = pd.read_sql_query(query, conn, params=[str(rider_id).strip()])
    conn.close()

    if row.empty:
        return None

    return row.iloc[0].to_dict()


def get_all_performance():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            p.*,
            COALESCE(r.rider_name, '') AS rider_name
        FROM performance p
        LEFT JOIN riders r ON p.rider_id = r.rider_id
        ORDER BY p.segment, p.rider_id
    """, conn)
    conn.close()
    return df


def to_int(value):
    try:
        if pd.isna(value) or value == "":
            return 0
        return int(float(value))
    except Exception:
        return 0


def to_float(value):
    try:
        if pd.isna(value) or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def pct(value):
    value = to_float(value)
    # Source Excel uses 1.0 = 100%
    if value <= 1.000001:
        value *= 100
    return f"{value:.2f}%"


def score_class(value):
    value = to_float(value)
    if value >= 0.95:
        return "excellent"
    if value >= 0.85:
        return "good"
    if value >= 0.70:
        return "warning"
    return "bad"


def segment_class(segment):
    s = str(segment).strip().upper()
    return {
        "A": "seg-a",
        "B": "seg-b",
        "C": "seg-c",
        "D": "seg-d",
        "E": "seg-e",
        "F": "seg-f",
    }.get(s, "seg-other")


def validate_performance_file(df):
    missing = [c for c in PERFORMANCE_COLUMNS if c not in df.columns]
    return missing


# ------------------------------------------------------------
# ADMIN AUTH
# ------------------------------------------------------------

def password_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def admin_password():
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except Exception:
        # Change this before production, or add ADMIN_PASSWORD
        # to Streamlit Secrets.
        return "ChangeMe123!"


def admin_login():
    if st.session_state.get("admin_ok"):
        return True

    st.markdown("## 🔐 Admin")
    st.caption("Enter the admin password to manage performance data.")

    with st.form("admin_login"):
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)

    if submit:
        if password_hash(password) == password_hash(admin_password()):
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


# ------------------------------------------------------------
# UI CSS
# ------------------------------------------------------------

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .brand {
        text-align:center;
        margin-bottom: 2rem;
    }

    .brand-icon {
        font-size: 46px;
        line-height: 1;
        margin-bottom: 8px;
    }

    .brand-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.7px;
    }

    .brand-subtitle {
        color:#6b7280;
        font-size:15px;
        margin-top:5px;
    }

    .lookup-card {
        background: linear-gradient(145deg,#ffffff,#f8fafc);
        border:1px solid #e5e7eb;
        border-radius:22px;
        padding:28px;
        box-shadow:0 10px 35px rgba(15,23,42,.07);
        margin-bottom:25px;
    }

    .profile {
        background: linear-gradient(145deg,#0f172a,#1e293b);
        color:white;
        border-radius:24px;
        padding:30px;
        margin-top:25px;
        box-shadow:0 15px 45px rgba(15,23,42,.18);
    }

    .profile-name {
        font-size:30px;
        font-weight:800;
    }

    .profile-id {
        color:#cbd5e1;
        margin-top:4px;
        font-size:14px;
    }

    .segment {
        display:inline-block;
        min-width:76px;
        text-align:center;
        border-radius:14px;
        padding:10px 18px;
        font-size:25px;
        font-weight:900;
        margin-top:15px;
    }

    .seg-a {background:#16a34a;color:#fff;}
    .seg-b {background:#2563eb;color:#fff;}
    .seg-c {background:#f59e0b;color:#fff;}
    .seg-d {background:#f97316;color:#fff;}
    .seg-e {background:#ef4444;color:#fff;}
    .seg-f {background:#7f1d1d;color:#fff;}
    .seg-other {background:#64748b;color:#fff;}

    .metric-card {
        background:white;
        border:1px solid #e5e7eb;
        border-radius:18px;
        padding:18px;
        min-height:110px;
        box-shadow:0 5px 18px rgba(15,23,42,.04);
    }

    .metric-label {
        color:#64748b;
        font-size:13px;
        font-weight:600;
        margin-bottom:8px;
    }

    .metric-value {
        color:#0f172a;
        font-size:25px;
        font-weight:800;
    }

    .metric-small {
        color:#64748b;
        font-size:12px;
        margin-top:5px;
    }

    .section-title {
        font-size:20px;
        font-weight:800;
        margin:30px 0 14px;
    }

    .info-line {
        background:#f8fafc;
        border:1px solid #e5e7eb;
        border-radius:13px;
        padding:12px 15px;
        margin-bottom:8px;
    }

    .admin-box {
        background:#f8fafc;
        border:1px solid #e2e8f0;
        border-radius:18px;
        padding:22px;
        margin-bottom:20px;
    }

    .footer-note {
        text-align:center;
        color:#94a3b8;
        font-size:12px;
        margin-top:35px;
    }

    @media (max-width: 700px) {
        .brand-title {font-size:26px;}
        .profile {padding:22px;}
        .profile-name {font-size:24px;}
    }
</style>
""", unsafe_allow_html=True)

init_db()


# ------------------------------------------------------------
# NAVIGATION / ADMIN
# ------------------------------------------------------------

# Keep the selected page in session state so Admin is always reachable
# from the main screen, even if the sidebar is collapsed.
if "page" not in st.session_state:
    st.session_state["page"] = "Rider Performance"

st.markdown("""
<style>
    .top-nav {
        display:flex;
        justify-content:center;
        gap:12px;
        margin:0 auto 25px auto;
        max-width:1050px;
    }
    .admin-access-note {
        text-align:center;
        color:#64748b;
        font-size:12px;
        margin-top:-15px;
        margin-bottom:20px;
    }
</style>
""", unsafe_allow_html=True)

nav1, nav2 = st.columns([3, 1])

with nav1:
    if st.button(
        "🏆 Rider Performance",
        use_container_width=True,
        type="primary" if st.session_state["page"] == "Rider Performance" else "secondary"
    ):
        st.session_state["page"] = "Rider Performance"
        st.rerun()

with nav2:
    if st.button(
        "⚙️ Admin",
        use_container_width=True,
        type="primary" if st.session_state["page"] == "Admin" else "secondary"
    ):
        st.session_state["page"] = "Admin"
        st.rerun()

page = st.session_state["page"]

if st.session_state.get("admin_ok") and page == "Admin":
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["admin_ok"] = False
        st.session_state["page"] = "Rider Performance"
        st.rerun()


# ------------------------------------------------------------
# PUBLIC RIDER PERFORMANCE
# ------------------------------------------------------------

if page == "Rider Performance":

    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🏆</div>
        <div class="brand-title">Rider Performance</div>
        <div class="brand-subtitle">Check your delivery performance</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="admin-access-note">⚙️ Admin access is available from the button above.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="lookup-card">', unsafe_allow_html=True)

    with st.form("rider_lookup"):
        rider_id = st.text_input(
            "Rider ID",
            placeholder="Enter your Rider ID",
            label_visibility="visible"
        ).strip()

        search = st.form_submit_button(
            "View My Performance",
            use_container_width=True,
            type="primary"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if search:
        if not rider_id:
            st.warning("Please enter your Rider ID.")
        else:
            data = get_performance(rider_id)

            if data is None:
                st.error("No performance record was found for this Rider ID.")
            else:
                name = data.get("rider_name", "").strip()
                if not name:
                    name = "Rider"

                segment = str(data.get("segment", "—")).strip()

                st.markdown(f"""
                <div class="profile">
                    <div class="profile-name">{html.escape(name)}</div>
                    <div class="profile-id">
                        Rider ID: {html.escape(str(data.get("rider_id", rider_id)))}
                    </div>
                    <div class="segment {segment_class(segment)}">
                        {html.escape(segment)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-title">📊 Performance Overview</div>',
                            unsafe_allow_html=True)

                cols = st.columns(3)

                cards = [
                    ("Verification Score", pct(data.get("verification_success_rate"))),
                    ("On-Time Delivery Score", pct(data.get("on_time_delivery_score"))),
                    ("Fail Rate Score", pct(data.get("fail_rate_score"))),
                    ("Final Delivery Quality Score", pct(data.get("final_delivery_quality_score"))),
                    ("Gross Orders", f"{to_int(data.get('gross_orders')):,}"),
                    ("Completed Orders", f"{to_int(data.get('completed_orders')):,}"),
                    ("Completed Orders In-Time", f"{to_int(data.get('completed_orders_in_time')):,}"),
                    (
                        "Late Orders",
                        f"{max(0, to_int(data.get('completed_orders')) - to_int(data.get('completed_orders_in_time'))):,}"
                    ),
                    ("Failed Orders", f"{to_int(data.get('failed_orders_by_rider')):,}"),
                ]

                for i, (label, value) in enumerate(cards):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{html.escape(label)}</div>
                            <div class="metric-value">{html.escape(value)}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('<div class="section-title">🔎 Full Performance Details</div>',
                            unsafe_allow_html=True)

                details = [
                    ("Month", data.get("month") or data.get("Month") or "—"),
                    ("City", data.get("city_name", "—")),
                    ("Contract", data.get("contract_name", "—")),
                    ("Vehicle", data.get("vehicle_type", "—")),
                    ("Verification Requests", f"{to_int(data.get('total_verification_requests')):,}"),
                    ("Successful Verification", f"{to_int(data.get('successful_verification_requests')):,}"),
                ]

                for label, value in details:
                    st.markdown(f"""
                    <div class="info-line">
                        <strong>{html.escape(label)}</strong>
                        <span style="float:right">{html.escape(str(value))}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    '<div class="footer-note">Performance is based on the latest uploaded report.</div>',
                    unsafe_allow_html=True
                )


# ------------------------------------------------------------
# ADMIN
# ------------------------------------------------------------

else:
    if not admin_login():
        st.stop()

    st.title("⚙️ Performance Admin")
    st.caption("Upload the latest performance report and manage rider names.")

    tabs = st.tabs([
        "📥 Upload Performance",
        "👤 Rider Names",
        "📊 Current Data"
    ])

    # --------------------------------------------------------
    # Upload performance
    # --------------------------------------------------------
    with tabs[0]:
        st.markdown('<div class="admin-box">', unsafe_allow_html=True)

        st.subheader("Upload Excel Performance Report")

        uploaded = st.file_uploader(
            "Excel file",
            type=["xlsx", "xls"],
            help="Upload the 3PL Delivery Quality Segmentation report."
        )

        if uploaded:
            try:
                df = pd.read_excel(uploaded)

                missing = validate_performance_file(df)

                if missing:
                    st.error("Missing required columns:")
                    st.code("\n".join(missing))
                else:
                    st.success(f"File loaded successfully — {len(df):,} rows found.")

                    st.dataframe(
                        df[PERFORMANCE_COLUMNS].head(20),
                        use_container_width=True,
                        hide_index=True
                    )

                    if st.button(
                        "💾 Import / Update Performance",
                        type="primary",
                        use_container_width=True
                    ):
                        clean = df[PERFORMANCE_COLUMNS].copy()

                        clean["rider_id"] = clean["rider_id"].astype(str).str.strip()

                        # Remove empty IDs
                        clean = clean[
                            (clean["rider_id"] != "") &
                            (clean["rider_id"].str.lower() != "nan")
                        ].copy()

                        save_performance(clean)

                        # If the Excel later contains rider_name,
                        # automatically save it as well.
                        if "rider_name" in df.columns:
                            name_df = df[["rider_id", "rider_name"]].copy()
                            name_df["rider_id"] = name_df["rider_id"].astype(str).str.strip()
                            name_df["rider_name"] = name_df["rider_name"].astype(str).str.strip()
                            save_names(name_df)

                        st.success(
                            f"✅ Performance updated successfully for {len(clean):,} riders."
                        )

            except Exception as e:
                st.error(f"Could not read the Excel file: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            "Your current report uses the exact performance columns from the uploaded "
            "3PL Delivery Quality Segmentation file."
        )

    # --------------------------------------------------------
    # Rider names
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("👤 Rider Names")

        st.caption(
            "The current performance Excel contains Rider ID but no rider name. "
            "Add the name here once; it will be shown on the public performance page."
        )

        with st.form("add_name"):
            c1, c2 = st.columns(2)
            rid = c1.text_input("Rider ID")
            rname = c2.text_input("Rider Name")

            save = st.form_submit_button(
                "Save Rider Name",
                type="primary",
                use_container_width=True
            )

        if save:
            if not rid.strip() or not rname.strip():
                st.error("Rider ID and Rider Name are required.")
            else:
                save_names(pd.DataFrame([{
                    "rider_id": rid.strip(),
                    "rider_name": rname.strip()
                }]))
                st.success("✅ Rider name saved.")

        st.divider()

        st.markdown("#### Bulk name upload (optional)")

        names_file = st.file_uploader(
            "Upload Excel/CSV with rider_id and rider_name",
            type=["xlsx", "xls", "csv"],
            key="names_file"
        )

        if names_file:
            try:
                if names_file.name.lower().endswith(".csv"):
                    names_df = pd.read_csv(names_file)
                else:
                    names_df = pd.read_excel(names_file)

                if "rider_id" not in names_df.columns or "rider_name" not in names_df.columns:
                    st.error("The file must contain: rider_id and rider_name")
                else:
                    st.dataframe(
                        names_df[["rider_id", "rider_name"]].head(20),
                        use_container_width=True,
                        hide_index=True
                    )

                    if st.button(
                        "Import Names",
                        type="primary",
                        use_container_width=True
                    ):
                        n = save_names(names_df[["rider_id", "rider_name"]])
                        st.success(f"✅ Saved {n:,} rider names.")

            except Exception as e:
                st.error(f"Could not read names file: {e}")

        current_names = load_names()

        if not current_names.empty:
            st.markdown("#### Saved names")
            st.dataframe(
                current_names,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # Current data
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("📊 Current Performance Data")

        current = get_all_performance()

        if current.empty:
            st.info("No performance data has been uploaded yet.")
        else:
            st.metric("Riders", f"{current['rider_id'].nunique():,}")

            show_cols = [
                "rider_id",
                "rider_name",
                "gross_orders",
                "completed_orders",
                "completed_orders_in_time",
                "failed_orders_by_rider",
                "on_time_delivery_score",
                "fail_rate_score",
                "final_delivery_quality_score",
                "segment",
            ]

            st.dataframe(
                current[show_cols],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Current Data",
                data=current.to_csv(index=False).encode("utf-8-sig"),
                file_name="rider_performance_current.csv",
                mime="text/csv",
                use_container_width=True
            )
