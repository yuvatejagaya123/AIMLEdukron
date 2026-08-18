import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features

from utils.charts import (
    histogram,
    scatter_chart
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Annuity Analysis",
    page_icon="💸",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("💸 Annuity Analysis")

st.markdown("""
Study customers' annual loan payment obligations
and understand their relationship with default risk.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# ANNUITY GROUPS
# =====================================================

df["ANNUITY_GROUP"] = pd.cut(
    df["AMT_ANNUITY"],
    bins=[
        0,
        10000,
        20000,
        30000,
        50000,
        70000,
        float("inf")
    ],
    labels=[
        "Below 10K",
        "10K-20K",
        "20K-30K",
        "30K-50K",
        "50K-70K",
        "Above 70K"
    ]
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Annuity Analysis Filters")

annuity_group_filter = st.sidebar.multiselect(
    "Annuity Group",
    options=df["ANNUITY_GROUP"].dropna().unique(),
    default=df["ANNUITY_GROUP"].dropna().unique()
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
    (df["ANNUITY_GROUP"].isin(annuity_group_filter))
    &
    (df["NAME_INCOME_TYPE"].isin(income_type_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CARDS
# =====================================================

average_annuity = filtered_df["AMT_ANNUITY"].mean()

median_annuity = filtered_df["AMT_ANNUITY"].median()

maximum_annuity = filtered_df["AMT_ANNUITY"].max()

average_annuity_defaulters = (
    filtered_df[
        filtered_df["TARGET"] == 1
    ]["AMT_ANNUITY"]
    .mean()
)

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Annuity",
    f"{average_annuity:,.0f}"
)

col2.metric(
    "Median Annuity",
    f"{median_annuity:,.0f}"
)

col3.metric(
    "Maximum Annuity",
    f"{maximum_annuity:,.0f}"
)

col4.metric(
    "Avg Annuity (Defaulters)",
    f"{average_annuity_defaulters:,.0f}"
)

st.divider()

# =====================================================
# ANNUITY DISTRIBUTION
# =====================================================

st.subheader("📊 Annuity Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "AMT_ANNUITY",
        "Annuity Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# ANNUITY BY TARGET
# =====================================================

annuity_target_df = (
    filtered_df.groupby("TARGET")["AMT_ANNUITY"]
    .mean()
    .reset_index()
)

fig_target = px.bar(
    annuity_target_df,
    x="TARGET",
    y="AMT_ANNUITY",
    text="AMT_ANNUITY",
    title="Annuity by TARGET"
)

st.plotly_chart(
    fig_target,
    use_container_width=True
)

st.divider()

# =====================================================
# SAMPLE DATA FOR SCATTER PLOTS
# =====================================================

scatter_df = filtered_df.sample(
    n=min(10000, len(filtered_df)),
    random_state=42
)

# =====================================================
# ANNUITY VS INCOME / CREDIT
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        scatter_chart(
            scatter_df,
            "AMT_INCOME_TOTAL",
            "AMT_ANNUITY",
            "TARGET",
            "Annuity vs Income"
        ),
        use_container_width=True
    )

with col2:

    st.plotly_chart(
        scatter_chart(
            scatter_df,
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "TARGET",
            "Annuity vs Credit"
        ),
        use_container_width=True
    )

st.divider()

# =====================================================
# AVG ANNUITY BY INCOME TYPE
# =====================================================

income_annuity_df = (
    filtered_df.groupby("NAME_INCOME_TYPE")
    ["AMT_ANNUITY"]
    .mean()
    .reset_index()
)

fig_income = px.bar(
    income_annuity_df,
    x="NAME_INCOME_TYPE",
    y="AMT_ANNUITY",
    title="Average Annuity by Income Type"
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY ANNUITY GROUP
# =====================================================

annuity_risk_df = (
    filtered_df.groupby(
        "ANNUITY_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .reset_index()
)

annuity_risk_df["Default Rate %"] = (
    annuity_risk_df["TARGET"] * 100
)

fig_risk = px.bar(
    annuity_risk_df,
    x="ANNUITY_GROUP",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Annuity Group"
)

fig_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

highest_risk_annuity_group = (
    annuity_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["ANNUITY_GROUP"]
)

st.subheader("💡 Annuity Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Average Annuity: {average_annuity:,.0f}

✅ Median Annuity: {median_annuity:,.0f}

✅ Maximum Annuity: {maximum_annuity:,.0f}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Average Annuity of Defaulters: {average_annuity_defaulters:,.0f}

⚠ Highest Risk Annuity Group: {highest_risk_annuity_group}
"""
    )
  
