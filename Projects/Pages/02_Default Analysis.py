import streamlit as st

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.filters import sidebar_filters

from utils.charts import (
    bar_chart,
    donut_chart
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Target / Default Analysis",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🎯 Target / Default Analysis")

st.markdown("""
This page focuses on the TARGET variable and helps understand
customer default behavior across different customer segments.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# FILTERS
# =====================================================

df = sidebar_filters(df)

# =====================================================
# KPI CALCULATIONS
# =====================================================

target_0 = (df["TARGET"] == 0).sum()

target_1 = (df["TARGET"] == 1).sum()

total_customers = len(df)

default_rate = (target_1 / total_customers) * 100

non_default_rate = (target_0 / total_customers) * 100


# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "TARGET = 0 Customers",
    f"{target_0:,}"
)

col2.metric(
    "TARGET = 1 Customers",
    f"{target_1:,}"
)

col3.metric(
    "Default Rate %",
    f"{default_rate:.2f}%"
)

col4.metric(
    "Non-Default Rate %",
    f"{non_default_rate:.2f}%"
)

st.divider()

# =====================================================
# TARGET COUNT CHART
# =====================================================

st.subheader("📊 TARGET Count Analysis")

target_count_df = (
    df["TARGET"]
    .value_counts()
    .reset_index()
)

target_count_df.columns = [
    "TARGET",
    "Count"
]

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        bar_chart(
            target_count_df,
            "TARGET",
            "Count",
            "TARGET Count Bar Chart"
        ),
        use_container_width=True
    )

with col2:

    st.plotly_chart(
        donut_chart(
            df,
            "TARGET",
            "TARGET Percentage Distribution"
        ),
        use_container_width=True
    )

st.divider()

# =====================================================
# DEFAULT RATE BY GENDER
# =====================================================

gender_default_df = (
    df.groupby("CODE_GENDER")["TARGET"]
    .mean()
    .reset_index()
)

gender_default_df["Default Rate %"] = (
    gender_default_df["TARGET"] * 100
)

# =====================================================
# DEFAULT RATE BY INCOME TYPE
# =====================================================

income_default_df = (
    df.groupby("NAME_INCOME_TYPE")["TARGET"]
    .mean()
    .reset_index()
)

income_default_df["Default Rate %"] = (
    income_default_df["TARGET"] * 100
)

st.subheader("👥 Default Rate by Customer Profile")

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        bar_chart(
            gender_default_df,
            "CODE_GENDER",
            "Default Rate %",
            "Default Rate by Gender"
        ),
        use_container_width=True
    )

with col2:

    st.plotly_chart(
        bar_chart(
            income_default_df,
            "NAME_INCOME_TYPE",
            "Default Rate %",
            "Default Rate by Income Type"
        ),
        use_container_width=True
    )

st.divider()

# =====================================================
# DEFAULT RATE BY EDUCATION
# =====================================================

education_default_df = (
    df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
    .mean()
    .reset_index()
)

education_default_df["Default Rate %"] = (
    education_default_df["TARGET"] * 100
)

# =====================================================
# DEFAULT RATE BY CONTRACT TYPE
# =====================================================

contract_default_df = (
    df.groupby("NAME_CONTRACT_TYPE")["TARGET"]
    .mean()
    .reset_index()
)

contract_default_df["Default Rate %"] = (
    contract_default_df["TARGET"] * 100
)

st.subheader("📚 Default Rate by Customer Segments")

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        bar_chart(
            education_default_df,
            "NAME_EDUCATION_TYPE",
            "Default Rate %",
            "Default Rate by Education"
        ),
        use_container_width=True
    )

with col2:

    st.plotly_chart(
        bar_chart(
            contract_default_df,
            "NAME_CONTRACT_TYPE",
            "Default Rate %",
            "Default Rate by Contract Type"
        ),
        use_container_width=True
    )

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

highest_risk_gender = (
    gender_default_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["CODE_GENDER"]
)

highest_risk_income = (
    income_default_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_INCOME_TYPE"]
)

highest_risk_education = (
    education_default_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_EDUCATION_TYPE"]
)

highest_risk_contract = (
    contract_default_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_CONTRACT_TYPE"]
)

st.subheader("💡 Key Risk Insights")

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
✅ Overall Default Rate : {default_rate:.2f}%

✅ Non-Default Rate : {non_default_rate:.2f}%

✅ Highest Risk Gender : {highest_risk_gender}
"""
    )

with col2:

    st.warning(
        f"""
⚠ Highest Risk Income Type : {highest_risk_income}

⚠ Highest Risk Education : {highest_risk_education}

⚠ Highest Risk Contract Type : {highest_risk_contract}
"""
    )

# =====================================================
# FORMULA
# =====================================================

with st.expander("🧮 Default Rate Formula"):

    st.markdown("""
### Formula

**Default Rate %**

= (Number of TARGET = 1 Customers ÷ Total Customers) × 100

This metric indicates the percentage of customers who experienced payment difficulties.
""")
