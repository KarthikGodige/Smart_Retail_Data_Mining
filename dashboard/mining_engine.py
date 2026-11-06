
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ==================================================
# 1. STANDARDIZE COLUMN NAMES
# ==================================================

def standardize_columns(df):

    df = df.copy()

    column_mapping = {
        "Invoice": "InvoiceNo",
        "Invoice No": "InvoiceNo",
        "Stock Code": "StockCode",
        "Customer ID": "CustomerID",
        "Price": "UnitPrice",
        "Description": "Description",
        "Quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Country": "Country"
    }

    df.columns = df.columns.str.strip()

    df = df.rename(columns=column_mapping)

    required_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


# ==================================================
# 2. CLEAN RETAIL DATA
# ==================================================

def clean_data(df):

    df = standardize_columns(df)

    df = df.copy()

    # Convert numeric columns
    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    # Remove missing essential values
    df = df.dropna(
        subset=[
            "InvoiceNo",
            "Description",
            "CustomerID",
            "InvoiceDate",
            "Quantity",
            "UnitPrice"
        ]
    )

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove cancelled invoices
    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    # Remove invalid transactions
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    # Calculate transaction amount
    df["TotalAmount"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    if df.empty:
        raise ValueError(
            "No valid retail transactions remain after cleaning."
        )

    return df


# ==================================================
# 3. RFM ANALYSIS
# ==================================================

def perform_rfm(df):

    reference_date = (
        df["InvoiceDate"].max() +
        pd.Timedelta(days=1)
    )

    rfm = df.groupby("CustomerID").agg({

        "InvoiceDate": lambda x: (
            reference_date - x.max()
        ).days,

        "InvoiceNo": "nunique",

        "TotalAmount": "sum"

    })

    rfm.columns = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    return rfm


# ==================================================
# 4. K-MEANS CLUSTERING
# ==================================================

def perform_clustering(rfm, n_clusters=5):
    if len(rfm) < n_clusters:
        raise ValueError("Not enough customers for clustering.")

    rfm_log = np.log1p(rfm)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm
# ==================================================
# 5. CUSTOMER SEGMENT SUMMARY
# ==================================================

def create_cluster_summary(rfm):

    summary = rfm.groupby("Cluster").agg({

        "Recency": "mean",

        "Frequency": "mean",

        "Monetary": "mean",

        "Cluster": "count"

    }).rename(columns={

        "Cluster": "CustomerCount"

    }).reset_index()

    return summary


# ==================================================
# 6. MARKET BASKET ANALYSIS
# ==================================================

def perform_market_basket(
    df,
    min_support=0.02,
    min_confidence=0.3
):
    uk_data = df[
        df["Country"].astype(str).str.strip().str.upper()
        == "UNITED KINGDOM"
    ].copy()

    if uk_data.empty:
        raise ValueError(
            "No United Kingdom transactions available "
            "for this market basket analysis."
        )

    basket = (
        uk_data.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    top_products = basket.sum().nlargest(300).index
    basket = basket[top_products]
    basket = basket.gt(0)

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=3
    )

    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return pd.DataFrame()

    rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(map(str, x)))
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(map(str, x)))
    )

    return rules


# ==================================================
# 7. COMPLETE DATA MINING PIPELINE
# ==================================================

def run_data_mining(
    uploaded_df,
    n_clusters=5,
    min_support=0.02,
    min_confidence=0.3
):
    cleaned_df = clean_data(uploaded_df)

    rfm = perform_rfm(cleaned_df)

    rfm = perform_clustering(
        rfm,
        n_clusters=n_clusters
    )

    cluster_summary = create_cluster_summary(rfm)

    rules = perform_market_basket(
        cleaned_df,
        min_support=min_support,
        min_confidence=min_confidence
    )

    return {
        "cleaned_data": cleaned_df,
        "customer_segments": rfm,
        "cluster_summary": cluster_summary,
        "association_rules": rules
    }