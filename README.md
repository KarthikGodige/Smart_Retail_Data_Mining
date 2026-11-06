# Smart Retail Customer Segmentation and Market Basket Analysis

An interactive retail analytics dashboard built using Python and Streamlit. This project applies data mining techniques to retail transaction data to identify customer segments, discover product associations, and generate actionable business insights.

## Project Overview

The project analyzes historical online retail transactions using three major data mining techniques:

- **RFM Analysis:** Evaluates customers based on Recency, Frequency, and Monetary value.
- **K-Means Clustering:** Groups customers into meaningful segments based on their purchasing behavior.
- **Apriori Algorithm:** Identifies frequently purchased product combinations and generates association rules.

The results are presented through an interactive Streamlit dashboard.

## Features

The dashboard includes five main sections:

1. **Overview** – Displays key retail business metrics and dataset information.
2. **Customer Segmentation** – Explores customer groups identified through RFM analysis and K-Means clustering.
3. **Market Basket Analysis** – Displays frequent itemsets and product association rules.
4. **Business Insights** – Presents findings and recommendations based on the analysis.
5. **Upload & Analyze** – Allows users to upload CSV or Excel datasets and perform data mining with configurable parameters.

## Data Mining Techniques

### 1. RFM Analysis

Customers are evaluated using:

- **Recency:** How recently a customer made a purchase.
- **Frequency:** How often a customer made purchases.
- **Monetary:** How much a customer spent.

### 2. K-Means Clustering

K-Means groups customers according to their RFM characteristics.

The implementation uses feature scaling and configurable cluster counts. The default configuration uses five clusters.

### 3. Apriori Algorithm

Market Basket Analysis discovers associations between products purchased together.

The dashboard allows users to configure minimum support and minimum confidence. Association rules are evaluated using measures such as support, confidence, and lift.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Mlxtend
- Matplotlib
- Seaborn
- OpenPyXL

## Project Structure

```text
Smart_Retail_Data_Mining/
│
├── dashboard/
│   ├── app.py
│   └── mining_engine.py
│
├── data/
│   ├── cleaned_retail.csv
│   ├── customer_segments.csv
│   ├── cluster_summary.csv
│   └── association_rules.csv
│
├── notebooks/
│   └── day1_retail_analysis.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses the Online Retail dataset, which contains historical retail transactions from 2010–2011.

The prepared dataset contains:

- 392,692 cleaned transaction records
- 4,338 unique customers
- 3,665 unique products
- Approximately £8.89 million in total revenue

Data preprocessing includes removing duplicates, handling missing values, excluding invalid quantities and prices, and calculating transaction amounts.

The original Excel dataset is not included in this repository. The cleaned dataset and precomputed analysis results are included for dashboard execution.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/KarthikGodige/Smart_Retail_Data_Mining.git
cd Smart_Retail_Data_Mining
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## Key Project Outcomes

- Identified customer groups using RFM analysis and K-Means clustering.
- Discovered product associations through Apriori.
- Developed an interactive dashboard for exploring retail data.
- Enabled users to upload datasets and adjust data mining parameters.

## Limitations

- The original dataset represents historical transactions from 2010–2011 and may not reflect current retail behavior.
- Customer segmentation depends on the selected features and clustering parameters.
- Association rules indicate purchasing relationships, not causation.
- Market Basket Analysis is performed on UK transactions in the current implementation.

## Future Improvements

- Add advanced customer behavior visualizations.
- Compare different clustering algorithms.
- Improve recommendation capabilities using association rules.
- Deploy the dashboard publicly.

## Author

**Karthik Godige**

- GitHub: [KarthikGodige](https://github.com/KarthikGodige)

---

*An academic Data Mining project focused on retail analytics, customer segmentation, and product association discovery.*