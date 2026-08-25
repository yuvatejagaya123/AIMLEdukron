import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.filters import sidebar_filters

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Download & Export Center",
    page_icon="📥",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📥 Download & Export Center")

st.markdown("""
Export customer datasets, default customers,
high-risk customers and dashboard summaries.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

df = clean_data(df)

df = create_features(df)

# =====================================================
# DERIVED FEATURES
# =====================================================

df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"]
    /
    df["AMT_INCOME_TOTAL"]
)

df["ANNUITY_INCOME_RATIO"] = (
    df["AMT_ANNUITY"]
    /
    df["AMT_INCOME_TOTAL"]
)

df["AVG_EXT_SCORE"] = (
    df[
        [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        ]
    ]
    .mean(axis=1)
)

# =====================================================
# FILTERS
# =====================================================

filtered_df = sidebar_filters(df)

# =====================================================
# HIGH RISK CUSTOMERS
# =====================================================

high_risk_customers = filtered_df[
    (
        filtered_df["CREDIT_INCOME_RATIO"] > 6
    )
    |
    (
        filtered_df["ANNUITY_INCOME_RATIO"] > 0.35
    )
    |
    (
        filtered_df["AVG_EXT_SCORE"] < 0.35
    )
]

# =====================================================
# DEFAULT CUSTOMERS
# =====================================================

default_customers = filtered_df[
    filtered_df["TARGET"] == 1
]

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 Export Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{len(df):,}"
)

col2.metric(
    "Filtered Customers",
    f"{len(filtered_df):,}"
)

col3.metric(
    "Default Customers",
    f"{len(default_customers):,}"
)

col4.metric(
    "High Risk Customers",
    f"{len(high_risk_customers):,}"
)

st.divider()

# =====================================================
# DATA TABLE
# =====================================================

st.subheader("📋 Filtered Applicant Records")

display_columns = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "NAME_EDUCATION_TYPE",
    "NAME_CONTRACT_TYPE"
]

display_columns = [
    col
    for col in display_columns
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# DOWNLOAD FILTERED CUSTOMERS
# =====================================================

st.subheader("⬇ Download Options")

filtered_csv = (
    filtered_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇ Download Filtered Customers",
    data=filtered_csv,
    file_name="filtered_customers.csv",
    mime="text/csv"
)

# =====================================================
# DOWNLOAD DEFAULT CUSTOMERS
# =====================================================

default_csv = (
    default_customers
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇ Download Default Customers",
    data=default_csv,
    file_name="default_customers.csv",
    mime="text/csv"
)

# =====================================================
# DOWNLOAD HIGH RISK CUSTOMERS
# =====================================================

high_risk_csv = (
    high_risk_customers
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇ Download High Risk Customers",
    data=high_risk_csv,
    file_name="high_risk_customers.csv",
    mime="text/csv"
)

# =====================================================
# SUMMARY CSV
# =====================================================

summary_df = pd.DataFrame({

    "Metric":[
        "Total Customers",
        "Filtered Customers",
        "Default Customers",
        "High Risk Customers",
        "Default Rate"
    ],

    "Value":[
        len(df),
        len(filtered_df),
        len(default_customers),
        len(high_risk_customers),
        round(
            filtered_df["TARGET"].mean()*100,
            2
        )
        if len(filtered_df) > 0
        else 0
    ]
})

summary_csv = (
    summary_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇ Download Summary CSV",
    data=summary_csv,
    file_name="dashboard_summary.csv",
    mime="text/csv"
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

st.subheader("💡 Export Insights")

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
✅ Total Customers: {len(df):,}

✅ Filtered Customers: {len(filtered_df):,}

✅ Default Customers: {len(default_customers):,}
"""
    )

with col2:

    st.warning(
        f"""
⚠ High Risk Customers: {len(high_risk_customers):,}

⚠ Export filtered datasets for
further machine-learning modeling
or business analysis.
"""
    )