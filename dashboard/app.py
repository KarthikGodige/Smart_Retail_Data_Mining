
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from io import BytesIO

from mining_engine import run_data_mining


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Retail Data Mining",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background-color: #101827;
    }

    [data-testid="stSidebar"] * {
        color: #e5eaf3;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #14213d;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #68758a;
        font-size: 14px;
        margin-bottom: 24px;
    }

    .section-title {
        color: #14213d;
        font-size: 21px;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e7ebf2;
        border-radius: 14px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 3px 12px rgba(20, 33, 61, 0.04);
    }

    .metric-label {
        color: #718096;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #14213d;
        font-size: 25px;
        font-weight: 800;
    }

    .metric-note {
        color: #8792a5;
        font-size: 11px;
        margin-top: 5px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e7ebf2;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .info-card h4 {
        color: #14213d;
        margin-top: 0;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #68758a;
        font-size: 13px;
        margin-bottom: 0;
    }

    .hero-card {
        background: linear-gradient(120deg, #14213d, #243b67);
        border-radius: 18px;
        padding: 28px;
        color: white;
        margin-bottom: 25px;
    }

    .hero-card h2 {
        color: white;
        font-size: 25px;
        margin-bottom: 8px;
    }

    .hero-card p {
        color: #d6deee;
        font-size: 14px;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7ebf2;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e7ebf2;
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    .stDownloadButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #929bad;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "mining_results": None,
    "uploaded_dataset_bytes": None,
    "uploaded_dataset_name": None,
    "active_dataset": "Original Online Retail Dataset",
    "active_source": "Original Dataset",
    "algorithm_settings": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_original_data():

    cleaned_data = pd.read_csv(
        DATA_DIR / "cleaned_retail.csv"
    )

    customer_segments = pd.read_csv(
        DATA_DIR / "customer_segments.csv",
        index_col=0
    )

    cluster_summary = pd.read_csv(
        DATA_DIR / "cluster_summary.csv",
        index_col=0
    )

    association_rules_data = pd.read_csv(
        DATA_DIR / "association_rules.csv"
    )

    cleaned_data["InvoiceDate"] = pd.to_datetime(
        cleaned_data["InvoiceDate"],
        errors="coerce"
    )

    return {
        "cleaned_data": cleaned_data,
        "customer_segments": customer_segments,
        "cluster_summary": cluster_summary,
        "association_rules": association_rules_data
    }


@st.cache_data
def read_uploaded_dataset(file_bytes, file_name):

    extension = Path(file_name).suffix.lower()

    if extension == ".csv":

        df = pd.read_csv(BytesIO(file_bytes))

    elif extension in [".xlsx", ".xls"]:

        sheets = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=None
        )

        valid_sheets = [
            sheet for sheet in sheets.values()
            if sheet is not None and not sheet.empty
        ]

        if not valid_sheets:
            raise ValueError("The Excel file contains no usable data.")

        df = pd.concat(
            valid_sheets,
            ignore_index=True
        )

    else:
        raise ValueError(
            "Unsupported file format. Upload CSV or Excel."
        )

    return df


# =========================================================
# DATASET MANAGEMENT
# =========================================================

def get_original_results():
    return load_original_data()


def get_active_results():

    if st.session_state.mining_results is not None:
        return st.session_state.mining_results

    return get_original_results()


def reset_to_original():

    st.session_state.mining_results = None
    st.session_state.uploaded_dataset_bytes = None
    st.session_state.uploaded_dataset_name = None
    st.session_state.active_dataset = "Original Online Retail Dataset"
    st.session_state.active_source = "Original Dataset"
    st.session_state.algorithm_settings = None


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_segments(segments):

    segments = segments.copy()

    if "Cluster" not in segments.columns:
        return segments

    segment_names = {
        0: "Recent Low-Value",
        1: "Loyal Customers",
        2: "At-Risk Customers",
        3: "Potential Loyal Customers",
        4: "VIP Customers"
    }

    segments["Segment"] = segments["Cluster"].map(
        segment_names
    ).fillna(
        segments["Cluster"].apply(
            lambda x: f"Cluster {x}"
        )
    )

    return segments


def prepare_rules(rules):

    rules = rules.copy()

    if rules.empty:
        return rules

    rules = rules.rename(
        columns={
            "antecedents": "Products Purchased Together",
            "consequents": "Associated Products",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift"
        }
    )

    if "Lift" in rules.columns:
        rules = rules.sort_values(
            by="Lift",
            ascending=False
        )

    return rules


def format_currency(value):

    return f"£{value:,.2f}"


def format_number(value):

    return f"{value:,.0f}"


def get_dataset_metrics(data):

    total_revenue = data["TotalAmount"].sum()

    total_customers = data["CustomerID"].nunique()

    total_products = data["Description"].nunique()

    total_transactions = data["InvoiceNo"].nunique()

    return {
        "revenue": total_revenue,
        "customers": total_customers,
        "products": total_products,
        "transactions": total_transactions
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 15px 0 22px 0;">
            <h2 style="color:white; margin-bottom:2px;">
                🛍️ Smart Retail
            </h2>
            <p style="color:#9caac0; font-size:12px;">
                Data Mining & Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "Select a page",
        [
            "Overview",
            "Customer Segmentation",
            "Market Basket Analysis",
            "Business Insights",
            "Upload & Analyze"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### Active Dataset")

    st.caption(st.session_state.active_dataset)

    st.caption(
        f"Source: {st.session_state.active_source}"
    )

    if st.session_state.mining_results is not None:

        st.success("Uploaded analysis active")

    else:

        st.info("Original dataset active")

    if st.button(
        "↩ Reset to Original Dataset",
        use_container_width=True
    ):

        reset_to_original()
        st.rerun()

    st.markdown("---")

    st.caption("Smart Retail Data Mining Project")
    st.caption("Customer Segmentation • Market Basket Analysis")


# =========================================================
# LOAD ACTIVE DATA
# =========================================================

try:

    results = get_active_results()

    active_data = results["cleaned_data"]

    active_segments = prepare_segments(
        results["customer_segments"]
    )

    active_summary = results["cluster_summary"]

    active_rules = prepare_rules(
        results["association_rules"]
    )

    metrics = get_dataset_metrics(active_data)

except Exception as e:

    st.error(
        f"Unable to load project data: {e}"
    )

    st.info(
        "Check that all required CSV files are present in the data folder."
    )

    st.stop()


# =========================================================
# PAGE HEADER
# =========================================================

PAGE_DESCRIPTIONS = {
    "Overview": "A high-level view of retail performance and customer activity.",
    "Customer Segmentation": "Understand customer behavior using RFM analysis and K-Means clustering.",
    "Market Basket Analysis": "Discover product associations using Apriori and association rules.",
    "Business Insights": "Convert data mining results into meaningful business decisions.",
    "Upload & Analyze": "Upload your own retail dataset and run configurable data mining algorithms."
}

st.markdown(
    f"""
    <div class="main-title">{page}</div>
    <div class="sub-title">{PAGE_DESCRIPTIONS[page]}</div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero-card">
            <h2>Smart Retail Analytics Dashboard</h2>
            <p>
                Explore customer segments, identify purchasing patterns,
                and discover actionable insights from retail transaction data.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Retail Performance Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Revenue",
            format_currency(metrics["revenue"])
        )

    with col2:
        st.metric(
            "Unique Customers",
            format_number(metrics["customers"])
        )

    with col3:
        st.metric(
            "Unique Products",
            format_number(metrics["products"])
        )

    with col4:
        st.metric(
            "Transactions",
            format_number(metrics["transactions"])
        )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Revenue by Country")

        country_revenue = (
            active_data.groupby("Country")["TotalAmount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        if not country_revenue.empty:

            st.bar_chart(country_revenue)

        else:

            st.info("Country revenue data is unavailable.")

    with right:

        st.markdown("#### Customer Segment Distribution")

        if "Segment" in active_segments.columns:

            segment_counts = (
                active_segments["Segment"]
                .value_counts()
            )

            st.bar_chart(segment_counts)

        else:

            st.info("Customer segment data is unavailable.")

    st.markdown("---")

    st.markdown("#### Dataset Information")

    info1, info2 = st.columns(2)

    with info1:

        st.markdown(
            f"""
            <div class="info-card">
                <h4>Active Dataset</h4>
                <p>{st.session_state.active_dataset}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:

        if "InvoiceDate" in active_data.columns:

            date_min = active_data["InvoiceDate"].min()
            date_max = active_data["InvoiceDate"].max()

            date_range = (
                f"{date_min:%d %b %Y} – {date_max:%d %b %Y}"
                if pd.notna(date_min) and pd.notna(date_max)
                else "Unavailable"
            )

        else:

            date_range = "Unavailable"

        st.markdown(
            f"""
            <div class="info-card">
                <h4>Transaction Date Range</h4>
                <p>{date_range}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Preview cleaned retail data"):

        st.dataframe(
            active_data.head(100),
            use_container_width=True
        )


# =========================================================
# PAGE 2 — CUSTOMER SEGMENTATION
# =========================================================

elif page == "Customer Segmentation":

    st.markdown(
        '<div class="section-title">RFM Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        RFM analysis evaluates customers using three important metrics:
        Recency, Frequency, and Monetary value.
        K-Means clustering groups customers with similar purchasing behavior.
        """
    )

    if "Cluster" not in active_segments.columns:

        st.warning("Cluster information is unavailable.")

    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Number of Clusters",
                active_segments["Cluster"].nunique()
            )

        with col2:
            st.metric(
                "Customers Analyzed",
                format_number(len(active_segments))
            )

        with col3:
            st.metric(
                "Average Customer Value",
                format_currency(
                    active_segments["Monetary"].mean()
                )
                if "Monetary" in active_segments.columns
                else "Unavailable"
            )

        st.markdown("---")

        left, right = st.columns(2)

        with left:

            st.markdown("#### Customer Distribution")

            segment_counts = (
                active_segments["Segment"]
                .value_counts()
            )

            st.bar_chart(segment_counts)

        with right:

            st.markdown("#### Average Monetary Value")

            if "Monetary" in active_segments.columns:

                monetary_by_segment = (
                    active_segments.groupby("Segment")["Monetary"]
                    .mean()
                    .sort_values(ascending=False)
                )

                st.bar_chart(monetary_by_segment)

        st.markdown("---")

        st.markdown("#### Cluster Summary")

        st.dataframe(
            active_summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Customer-Level Segmentation")

        available_segments = sorted(
            active_segments["Segment"].dropna().unique()
        )

        selected_segment = st.selectbox(
            "Filter by customer segment",
            ["All Segments"] + list(available_segments)
        )

        filtered_segments = active_segments.copy()

        if selected_segment != "All Segments":

            filtered_segments = filtered_segments[
                filtered_segments["Segment"] == selected_segment
            ]

        st.dataframe(
            filtered_segments.head(500),
            use_container_width=True
        )

        st.download_button(
            "Download Customer Segments",
            data=filtered_segments.to_csv(index=True).encode("utf-8"),
            file_name="customer_segments.csv",
            mime="text/csv"
        )


# =========================================================
# PAGE 3 — MARKET BASKET ANALYSIS
# =========================================================

elif page == "Market Basket Analysis":

    st.markdown(
        '<div class="section-title">Market Basket Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Market Basket Analysis discovers products that frequently
        occur together in customer transactions using the Apriori algorithm.
        """
    )

    if active_rules.empty:

        st.warning(
            "No association rules are available for this dataset "
            "with the current analysis settings."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Association Rules",
                format_number(len(active_rules))
            )

        with col2:

            avg_confidence = (
                active_rules["Confidence"].mean()
                if "Confidence" in active_rules.columns
                else 0
            )

            st.metric(
                "Average Confidence",
                f"{avg_confidence:.2%}"
            )

        with col3:

            avg_lift = (
                active_rules["Lift"].mean()
                if "Lift" in active_rules.columns
                else 0
            )

            st.metric(
                "Average Lift",
                f"{avg_lift:.2f}"
            )

        st.markdown("---")

        st.markdown("#### Top Product Associations")

        display_columns = [
            col for col in [
                "Products Purchased Together",
                "Associated Products",
                "Support",
                "Confidence",
                "Lift"
            ]
            if col in active_rules.columns
        ]

        st.dataframe(
            active_rules[display_columns].head(20),
            use_container_width=True,
            hide_index=True
        )

        if "Lift" in active_rules.columns:

            st.markdown("#### Top Associations by Lift")

            top_rules = active_rules.head(10).copy()

            if (
                "Products Purchased Together" in top_rules.columns
                and "Associated Products" in top_rules.columns
            ):

                top_rules["Rule"] = (
                    top_rules["Products Purchased Together"]
                    .astype(str)
                    + " → "
                    + top_rules["Associated Products"].astype(str)
                )

                lift_chart = top_rules.set_index("Rule")["Lift"]

                st.bar_chart(lift_chart)

        st.download_button(
            "Download Association Rules",
            data=active_rules.to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv"
        )


# =========================================================
# PAGE 4 — BUSINESS INSIGHTS
# =========================================================

elif page == "Business Insights":

    st.markdown(
        '<div class="section-title">Business Intelligence & Insights</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        These insights translate the data mining results into
        practical recommendations for retail businesses.
        """
    )

    # Customer segment insight

    if (
        "Segment" in active_segments.columns
        and not active_segments.empty
    ):

        segment_counts = active_segments["Segment"].value_counts()

        largest_segment = segment_counts.idxmax()

        largest_count = segment_counts.max()

        st.markdown(
            f"""
            <div class="info-card">
                <h4>👥 Largest Customer Segment</h4>
                <p>
                    <b>{largest_segment}</b> is the largest segment,
                    containing {largest_count:,} customers.
                    Retailers can design targeted campaigns for this group.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # High-value customers

    if (
        "Monetary" in active_segments.columns
        and not active_segments.empty
    ):

        highest_value_segment = (
            active_segments.groupby("Segment")["Monetary"]
            .mean()
            .idxmax()
        )

        highest_value = (
            active_segments.groupby("Segment")["Monetary"]
            .mean()
            .max()
        )

        st.markdown(
            f"""
            <div class="info-card">
                <h4>💎 Highest Value Customer Segment</h4>
                <p>
                    <b>{highest_value_segment}</b> has the highest
                    average customer spending at
                    {format_currency(highest_value)}.
                    Loyalty rewards and personalized offers may help retain these customers.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Frequency insight

    if (
        "Frequency" in active_segments.columns
        and not active_segments.empty
    ):

        frequent_segment = (
            active_segments.groupby("Segment")["Frequency"]
            .mean()
            .idxmax()
        )

        st.markdown(
            f"""
            <div class="info-card">
                <h4>🔁 Most Frequent Customers</h4>
                <p>
                    <b>{frequent_segment}</b> has the highest
                    average purchase frequency.
                    Consider loyalty programs and repeat-purchase incentives.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Market basket insight

    if not active_rules.empty and "Lift" in active_rules.columns:

        top_rule = active_rules.iloc[0]

        antecedent = top_rule.get(
            "Products Purchased Together",
            "Product association"
        )

        consequent = top_rule.get(
            "Associated Products",
            "Associated product"
        )

        lift_value = top_rule["Lift"]

        st.markdown(
            f"""
            <div class="info-card">
                <h4>🛒 Strongest Product Association</h4>
                <p>
                    <b>{antecedent}</b> is associated with
                    <b>{consequent}</b>.
                    The lift value is <b>{lift_value:.2f}</b>.
                    This relationship may help retailers consider
                    product bundling and cross-selling opportunities.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Market basket insights are unavailable because no association rules were generated."
        )

    st.markdown("---")

    st.markdown("#### Recommended Business Strategies")

    recommendations = [
        (
            "Customer Retention",
            "Create loyalty programs for frequent and high-value customers."
        ),
        (
            "Customer Reactivation",
            "Offer personalized discounts to customers who have not purchased recently."
        ),
        (
            "Cross-Selling",
            "Use product associations to recommend complementary products."
        ),
        (
            "Personalized Marketing",
            "Create separate campaigns for different customer segments."
        ),
        (
            "Inventory Planning",
            "Use popular product combinations to support inventory and promotional decisions."
        )
    ]

    for title, description in recommendations:

        st.markdown(
            f"""
            <div class="info-card">
                <h4>{title}</h4>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PAGE 5 — UPLOAD & ANALYZE
# =========================================================

elif page == "Upload & Analyze":

    st.markdown(
        '<div class="section-title">Upload Your Retail Dataset</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Upload a retail transaction dataset in CSV or Excel format.
        The system will clean the data, perform RFM analysis,
        apply K-Means clustering, and discover product associations.
        """
    )

    st.info(
        """
        Required columns: InvoiceNo (or Invoice), StockCode,
        Description, Quantity, InvoiceDate, UnitPrice (or Price),
        CustomerID (or Customer ID), and Country.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Excel files with multiple worksheets are supported."
    )

    # Save uploaded file in session state

    if uploaded_file is not None:

        st.session_state.uploaded_dataset_bytes = uploaded_file.getvalue()

        st.session_state.uploaded_dataset_name = uploaded_file.name

    # Load and preview upload

    uploaded_df = None

    if st.session_state.uploaded_dataset_bytes is not None:

        try:

            uploaded_df = read_uploaded_dataset(
                st.session_state.uploaded_dataset_bytes,
                st.session_state.uploaded_dataset_name
            )

            st.success(
                f"Dataset loaded: {st.session_state.uploaded_dataset_name}"
            )

            st.markdown("#### Dataset Preview")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Rows",
                    format_number(len(uploaded_df))
                )

            with col2:
                st.metric(
                    "Total Columns",
                    len(uploaded_df.columns)
                )

            with col3:
                st.metric(
                    "Missing Values",
                    format_number(
                        uploaded_df.isna().sum().sum()
                    )
                )

            with st.expander("View uploaded dataset"):

                st.dataframe(
                    uploaded_df.head(100),
                    use_container_width=True
                )

            with st.expander("Column Information"):

                column_info = pd.DataFrame({
                    "Column": uploaded_df.columns,
                    "Data Type": uploaded_df.dtypes.astype(str).values,
                    "Missing Values": uploaded_df.isna().sum().values
                })

                st.dataframe(
                    column_info,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(f"Unable to read uploaded dataset: {e}")

            uploaded_df = None

    st.markdown("---")

    # Algorithm configuration

    st.markdown(
        '<div class="section-title">Algorithm Configuration</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Adjust the parameters below to experiment with different
        clustering and association-rule results.
        """
    )

    config1, config2, config3 = st.columns(3)

    with config1:

        n_clusters = st.slider(
            "Number of Customer Clusters",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Number of customer groups created by K-Means."
        )

    with config2:

        min_support = st.slider(
            "Minimum Support",
            min_value=0.01,
            max_value=0.10,
            value=0.02,
            step=0.01,
            format="%.2f",
            help="Minimum fraction of transactions containing an itemset."
        )

    with config3:

        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.10,
            max_value=1.00,
            value=0.30,
            step=0.05,
            format="%.2f",
            help="Minimum confidence required for an association rule."
        )

    st.caption(
        "Higher support and confidence thresholds generally produce fewer association rules."
    )

    st.markdown("---")

    # Execute mining

    if st.button(
        "🚀 Run Data Mining",
        type="primary",
        use_container_width=True,
        disabled=uploaded_df is None
    ):

        try:

            with st.spinner(
                "Cleaning data and running data mining algorithms. Please wait..."
            ):

                mining_results = run_data_mining(
                    uploaded_df,
                    n_clusters=n_clusters,
                    min_support=min_support,
                    min_confidence=min_confidence
                )

                st.session_state.mining_results = mining_results

                st.session_state.active_dataset = (
                    st.session_state.uploaded_dataset_name
                )

                st.session_state.active_source = "Uploaded Dataset"

                st.session_state.algorithm_settings = {
                    "n_clusters": n_clusters,
                    "min_support": min_support,
                    "min_confidence": min_confidence
                }

            st.success(
                "Data mining completed successfully!"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Data mining failed: {e}"
            )

    # Display previous analysis results

    if st.session_state.mining_results is not None:

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Latest Analysis Results</div>',
            unsafe_allow_html=True
        )

        latest = st.session_state.mining_results

        result_data = latest["cleaned_data"]
        result_segments = latest["customer_segments"]
        result_summary = latest["cluster_summary"]
        result_rules = latest["association_rules"]

        result_metrics = get_dataset_metrics(result_data)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Cleaned Transactions",
                format_number(len(result_data))
            )

        with c2:
            st.metric(
                "Customers",
                format_number(result_metrics["customers"])
            )

        with c3:
            st.metric(
                "Customer Clusters",
                result_segments["Cluster"].nunique()
            )

        with c4:
            st.metric(
                "Association Rules",
                format_number(len(result_rules))
            )

        if st.session_state.algorithm_settings:

            settings = st.session_state.algorithm_settings

            st.caption(
                f"Settings used: Clusters = {settings['n_clusters']} | "
                f"Support = {settings['min_support']:.2f} | "
                f"Confidence = {settings['min_confidence']:.2f}"
            )

        st.markdown("#### Customer Cluster Summary")

        st.dataframe(
            result_summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Association Rules Preview")

        if not result_rules.empty:

            display_rules = prepare_rules(result_rules)

            display_columns = [
                col for col in [
                    "Products Purchased Together",
                    "Associated Products",
                    "Support",
                    "Confidence",
                    "Lift"
                ]
                if col in display_rules.columns
            ]

            st.dataframe(
                display_rules[display_columns].head(10),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No association rules were generated with these settings. "
                "Try lowering minimum support or minimum confidence."
            )

        st.markdown("#### Download Analysis Results")

        download1, download2, download3 = st.columns(3)

        with download1:

            st.download_button(
                "Download Cleaned Data",
                data=result_data.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_retail.csv",
                mime="text/csv",
                use_container_width=True
            )

        with download2:

            st.download_button(
                "Download Customer Segments",
                data=result_segments.to_csv(index=True).encode("utf-8"),
                file_name="customer_segments.csv",
                mime="text/csv",
                use_container_width=True
            )

        with download3:

            st.download_button(
                "Download Association Rules",
                data=prepare_rules(result_rules).to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="association_rules.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Smart Retail Customer Segmentation and Market Basket Analysis
        <br>
        Data Mining Course End Project
    </div>
    """,
    unsafe_allow_html=True
)
