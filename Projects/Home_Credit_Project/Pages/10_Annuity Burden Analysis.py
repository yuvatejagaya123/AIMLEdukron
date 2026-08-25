import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.charts import (
    histogram,
    bar_chart
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Annuity Burden Analysis",
    page_icon="📉",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📉 Annuity Burden Analysis")

st.markdown("""
Analyze repayment burden based on the ratio between
loan annuity and customer income.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# ANNUITY INCOME RATIO
# =====================================================

df["ANNUITY_INCOME_RATIO"] = (
    df["AMT_ANNUITY"] /
    df["AMT_INCOME_TOTAL"]
)

# =====================================================
# RISK GROUPS
# =====================================================

df["BURDEN_GROUP"] = pd.cut(
    df["ANNUITY_INCOME_RATIO"],
    bins=[
        0,
        0.10,
        0.20,
        0.35,
        float("inf")
    ],
    labels=[
        "Low Repayment Burden",
        "Medium Repayment Burden",
        "High Repayment Burden",
        "Very High Repayment Burden"
    ]
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Burden Analysis Filters")

burden_filter = st.sidebar.multiselect(
    "Repayment Burden",
    options=df["BURDEN_GROUP"].dropna().unique(),
    default=df["BURDEN_GROUP"].dropna().unique()
)

income_type_filter = st.sidebar.multiselect(
    "Income Type",
    options=df["NAME_INCOME_TYPE"].dropna().unique(),
    default=df["NAME_INCOME_TYPE"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["BURDEN_GROUP"].isin(burden_filter))
    &
    (df["NAME_INCOME_TYPE"].isin(income_type_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CARDS
# =====================================================

avg_ratio = (
    filtered_df["ANNUITY_INCOME_RATIO"]
    .mean()
)

max_ratio = (
    filtered_df["ANNUITY_INCOME_RATIO"]
    .max()
)

high_burden_df = filtered_df[
    filtered_df["BURDEN_GROUP"].isin(
        [
            "High Repayment Burden",
            "Very High Repayment Burden"
        ]
    )
]

high_burden_default_rate = (
    high_burden_df["TARGET"]
    .mean()
) * 100

st.subheader("📌 KPI Cards")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Annuity-To-Income Ratio",
    f"{avg_ratio:.3f}"
)

col2.metric(
    "Highest Ratio",
    f"{max_ratio:.3f}"
)

col3.metric(
    "Default Rate (High Burden)",
    f"{high_burden_default_rate:.2f}%"
)

st.divider()

# =====================================================
# 1. ANNUITY TO INCOME DISTRIBUTION
# =====================================================

st.subheader("📊 Annuity-To-Income Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "ANNUITY_INCOME_RATIO",
        "Annuity-To-Income Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# 2. DEFAULT RATE BY BURDEN
# =====================================================

burden_risk_df = (
    filtered_df.groupby(
        "BURDEN_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .reset_index()
)

burden_risk_df["Default Rate %"] = (
    burden_risk_df["TARGET"] * 100
)

fig_burden_risk = px.bar(
    burden_risk_df,
    x="BURDEN_GROUP",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Repayment Burden"
)

fig_burden_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_burden_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# 3. RATIO BY GENDER
# =====================================================

gender_ratio_df = (
    filtered_df.groupby(
        "CODE_GENDER"
    )["ANNUITY_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_gender_ratio = px.bar(
    gender_ratio_df,
    x="CODE_GENDER",
    y="ANNUITY_INCOME_RATIO",
    text="ANNUITY_INCOME_RATIO",
    title="Ratio by Gender"
)

st.plotly_chart(
    fig_gender_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# 4. RATIO BY INCOME TYPE
# =====================================================

income_ratio_df = (
    filtered_df.groupby(
        "NAME_INCOME_TYPE"
    )["ANNUITY_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_income_ratio = px.bar(
    income_ratio_df,
    x="NAME_INCOME_TYPE",
    y="ANNUITY_INCOME_RATIO",
    title="Ratio by Income Type"
)

st.plotly_chart(
    fig_income_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# 5. RATIO BY EDUCATION
# =====================================================

education_ratio_df = (
    filtered_df.groupby(
        "NAME_EDUCATION_TYPE"
    )["ANNUITY_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_education_ratio = px.bar(
    education_ratio_df,
    x="NAME_EDUCATION_TYPE",
    y="ANNUITY_INCOME_RATIO",
    title="Ratio by Education"
)

st.plotly_chart(
    fig_education_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# 6. RATIO VS TARGET
# =====================================================

ratio_target_df = (
    filtered_df.groupby(
        "TARGET"
    )["ANNUITY_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_target_ratio = px.bar(
    ratio_target_df,
    x="TARGET",
    y="ANNUITY_INCOME_RATIO",
    text="ANNUITY_INCOME_RATIO",
    title="Ratio vs TARGET"
)

st.plotly_chart(
    fig_target_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

highest_risk_burden = (
    burden_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["BURDEN_GROUP"]
)

lowest_risk_burden = (
    burden_risk_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["BURDEN_GROUP"]
)

most_common_burden = (
    filtered_df["BURDEN_GROUP"]
    .value_counts()
    .idxmax()
)

st.subheader("💡 Annuity Burden Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Average Ratio: {avg_ratio:.3f}

✅ Highest Ratio: {max_ratio:.3f}

✅ Most Common Burden Group: {most_common_burden}

✅ Lowest Risk Burden Group: {lowest_risk_burden}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Burden Group: {highest_risk_burden}

⚠ Default Rate (High Burden): {high_burden_default_rate:.2f}%

⚠ Customers with high repayment burden may have a greater probability of payment difficulty.
"""
    )