import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Regional Risk Analysis",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🌍 Regional Risk Analysis")

st.markdown("""
Analyze regional characteristics and understand
their relationship with default behavior.
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

st.sidebar.header("Regional Risk Filters")

region_filter = st.sidebar.multiselect(
    "Region Rating",
    options=sorted(
        df["REGION_RATING_CLIENT"]
        .dropna()
        .unique()
    ),
    default=sorted(
        df["REGION_RATING_CLIENT"]
        .dropna()
        .unique()
    )
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["REGION_RATING_CLIENT"].isin(region_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

most_common_region_rating = (
    filtered_df["REGION_RATING_CLIENT"]
    .mode()[0]
)

region_risk_df = (
    filtered_df.groupby(
        "REGION_RATING_CLIENT"
    )["TARGET"]
    .mean()
    .reset_index()
)

region_risk_df["Default Rate %"] = (
    region_risk_df["TARGET"] * 100
)

highest_risk_region = (
    region_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["REGION_RATING_CLIENT"]
)

avg_region_population = (
    filtered_df["REGION_POPULATION_RELATIVE"]
    .mean()
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Most Common Region Rating",
    str(most_common_region_rating)
)

col2.metric(
    "Highest Risk Region Rating",
    str(highest_risk_region)
)

col3.metric(
    "Average Regional Population",
    f"{avg_region_population:.4f}"
)

st.divider()

# =====================================================
# CUSTOMERS BY REGION RATING
# =====================================================

region_count_df = (
    filtered_df["REGION_RATING_CLIENT"]
    .value_counts()
    .reset_index()
)

region_count_df.columns = [
    "Region Rating",
    "Customers"
]

fig_region_count = px.bar(
    region_count_df,
    x="Region Rating",
    y="Customers",
    text="Customers",
    title="Customers by Region Rating"
)

st.plotly_chart(
    fig_region_count,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY REGION RATING
# =====================================================

fig_region_risk = px.bar(
    region_risk_df,
    x="REGION_RATING_CLIENT",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Region Rating"
)

fig_region_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_region_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT BY REGION RATING
# =====================================================

region_credit_df = (
    filtered_df.groupby(
        "REGION_RATING_CLIENT"
    )["AMT_CREDIT"]
    .mean()
    .reset_index()
)

fig_region_credit = px.bar(
    region_credit_df,
    x="REGION_RATING_CLIENT",
    y="AMT_CREDIT",
    text="AMT_CREDIT",
    title="Credit by Region Rating"
)

fig_region_credit.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig_region_credit,
    use_container_width=True
)

st.divider()

# =====================================================
# INCOME BY REGION RATING
# =====================================================

region_income_df = (
    filtered_df.groupby(
        "REGION_RATING_CLIENT"
    )["AMT_INCOME_TOTAL"]
    .mean()
    .reset_index()
)

fig_region_income = px.bar(
    region_income_df,
    x="REGION_RATING_CLIENT",
    y="AMT_INCOME_TOTAL",
    text="AMT_INCOME_TOTAL",
    title="Income by Region Rating"
)

fig_region_income.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig_region_income,
    use_container_width=True
)

st.divider()

# =====================================================
# REGION MISMATCH VS DEFAULT
# =====================================================

filtered_df["REGION_MISMATCH"] = (
    (
        filtered_df["REG_REGION_NOT_LIVE_REGION"] == 1
    )
    |
    (
        filtered_df["REG_REGION_NOT_WORK_REGION"] == 1
    )
).astype(int)

region_mismatch_df = (
    filtered_df.groupby(
        "REGION_MISMATCH"
    )["TARGET"]
    .mean()
    .reset_index()
)

region_mismatch_df["Default Rate %"] = (
    region_mismatch_df["TARGET"] * 100
)

fig_region_mismatch = px.bar(
    region_mismatch_df,
    x="REGION_MISMATCH",
    y="Default Rate %",
    text="Default Rate %",
    title="Region Mismatch vs Default"
)

fig_region_mismatch.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_region_mismatch.update_layout(
    xaxis_title="Region Mismatch (0=No, 1=Yes)"
)

st.plotly_chart(
    fig_region_mismatch,
    use_container_width=True
)

st.divider()

# =====================================================
# CITY MISMATCH VS DEFAULT
# =====================================================

filtered_df["CITY_MISMATCH"] = (
    (
        filtered_df["REG_CITY_NOT_LIVE_CITY"] == 1
    )
    |
    (
        filtered_df["REG_CITY_NOT_WORK_CITY"] == 1
    )
).astype(int)

city_mismatch_df = (
    filtered_df.groupby(
        "CITY_MISMATCH"
    )["TARGET"]
    .mean()
    .reset_index()
)

city_mismatch_df["Default Rate %"] = (
    city_mismatch_df["TARGET"] * 100
)

fig_city_mismatch = px.bar(
    city_mismatch_df,
    x="CITY_MISMATCH",
    y="Default Rate %",
    text="Default Rate %",
    title="City Mismatch vs Default"
)

fig_city_mismatch.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_city_mismatch.update_layout(
    xaxis_title="City Mismatch (0=No, 1=Yes)"
)

st.plotly_chart(
    fig_city_mismatch,
    use_container_width=True
)

st.divider()

# =====================================================
# REGIONAL SUMMARY TABLE
# =====================================================

st.subheader("📋 Regional Summary")

regional_summary = (
    filtered_df.groupby(
        "REGION_RATING_CLIENT"
    )
    .agg(
        Customers=("SK_ID_CURR", "count"),
        Avg_Income=("AMT_INCOME_TOTAL", "mean"),
        Avg_Credit=("AMT_CREDIT", "mean"),
        Default_Rate=("TARGET", "mean")
    )
    .reset_index()
)

regional_summary["Default_Rate"] = (
    regional_summary["Default_Rate"] * 100
).round(2)

regional_summary["Avg_Income"] = (
    regional_summary["Avg_Income"]
).round(0)

regional_summary["Avg_Credit"] = (
    regional_summary["Avg_Credit"]
).round(0)

st.dataframe(
    regional_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

lowest_risk_region = (
    region_risk_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["REGION_RATING_CLIENT"]
)

region_mismatch_default_rate = (
    region_mismatch_df[
        region_mismatch_df["REGION_MISMATCH"] == 1
    ]["Default Rate %"]
).iloc[0]

city_mismatch_default_rate = (
    city_mismatch_df[
        city_mismatch_df["CITY_MISMATCH"] == 1
    ]["Default Rate %"]
).iloc[0]

st.subheader("💡 Regional Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Most Common Region Rating: {most_common_region_rating}

✅ Lowest Risk Region Rating: {lowest_risk_region}

✅ Average Regional Population Indicator: {avg_region_population:.4f}

✅ Regional profile influences customer financial characteristics.
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Region Rating: {highest_risk_region}

⚠ Region Mismatch Default Rate: {region_mismatch_default_rate:.2f}%

⚠ City Mismatch Default Rate: {city_mismatch_default_rate:.2f}%

⚠ Customers living and working in different locations may exhibit different repayment behavior.
"""
    )