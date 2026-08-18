import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Correlation & Risk Factor Analysis",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📈 Correlation & Risk Factor Analysis")

st.markdown("""
Identify important numerical relationships associated with loan default.
This page focuses on correlation with the TARGET variable and key financial,
demographic, employment and external score indicators.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

df = clean_data(df)

df = create_features(df)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
    365243,
    np.nan
)

df["AGE"] = (
    abs(df["DAYS_BIRTH"]) / 365
)

df["EMPLOYMENT_YEARS"] = (
    abs(df["DAYS_EMPLOYED"]) / 365
)

df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
)

df["ANNUITY_INCOME_RATIO"] = (
    df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
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

st.sidebar.header("Correlation Analysis Filters")

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].dropna().unique(),
    default=df["TARGET"].dropna().unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["CODE_GENDER"].dropna().unique(),
    default=df["CODE_GENDER"].dropna().unique()
)

education_filter = st.sidebar.multiselect(
    "Education",
    options=df["NAME_EDUCATION_TYPE"].dropna().unique(),
    default=df["NAME_EDUCATION_TYPE"].dropna().unique()
)

filtered_df = df[
    (df["TARGET"].isin(target_filter))
    &
    (df["CODE_GENDER"].isin(gender_filter))
    &
    (df["NAME_EDUCATION_TYPE"].isin(education_filter))
].copy()

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# =====================================================
# NUMERICAL FEATURES
# =====================================================

numeric_features = [
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AGE",
    "EMPLOYMENT_YEARS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AVG_EXT_SCORE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "REGION_RATING_CLIENT"
]

available_features = [
    col for col in numeric_features
    if col in filtered_df.columns
]

corr_df = filtered_df[available_features].copy()

corr_df = corr_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 Correlation Overview")

valid_corr_df = corr_df.dropna(
    subset=["TARGET"]
)

corr_matrix = valid_corr_df.corr(numeric_only=True)

target_corr = (
    corr_matrix["TARGET"]
    .drop("TARGET")
    .dropna()
    .sort_values()
)

strongest_positive_feature = (
    target_corr
    .sort_values(ascending=False)
    .index[0]
)

strongest_positive_value = (
    target_corr
    .sort_values(ascending=False)
    .iloc[0]
)

strongest_negative_feature = (
    target_corr
    .sort_values(ascending=True)
    .index[0]
)

strongest_negative_value = (
    target_corr
    .sort_values(ascending=True)
    .iloc[0]
)

avg_external_score = (
    filtered_df["AVG_EXT_SCORE"]
    .mean()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Numerical Features Used",
    len(available_features)
)

col2.metric(
    "Strongest Positive Corr",
    strongest_positive_feature
)

col3.metric(
    "Strongest Negative Corr",
    strongest_negative_feature
)

col4.metric(
    "Average External Score",
    f"{avg_external_score:.3f}"
)

st.divider()

# =====================================================
# 1. CORRELATION HEATMAP
# =====================================================

st.subheader("🔥 Correlation Heatmap")

fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Correlation Heatmap"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)

st.divider()

# =====================================================
# 2. CORRELATION WITH TARGET
# =====================================================

st.subheader("🎯 Correlation with TARGET")

target_corr_df = (
    target_corr
    .reset_index()
)

target_corr_df.columns = [
    "Feature",
    "Correlation with TARGET"
]

target_corr_df["Abs Correlation"] = (
    target_corr_df["Correlation with TARGET"]
    .abs()
)

target_corr_df = (
    target_corr_df
    .sort_values(
        "Abs Correlation",
        ascending=False
    )
)

fig_target_corr = px.bar(
    target_corr_df,
    x="Feature",
    y="Correlation with TARGET",
    text="Correlation with TARGET",
    title="Correlation with TARGET"
)

fig_target_corr.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

st.plotly_chart(
    fig_target_corr,
    use_container_width=True
)

st.divider()

# =====================================================
# 3. TOP POSITIVE AND NEGATIVE CORRELATIONS
# =====================================================

st.subheader("📊 Top Positive and Negative Correlations")

positive_corr_df = (
    target_corr_df[
        target_corr_df["Correlation with TARGET"] > 0
    ]
    .sort_values(
        "Correlation with TARGET",
        ascending=False
    )
    .head(10)
)

negative_corr_df = (
    target_corr_df[
        target_corr_df["Correlation with TARGET"] < 0
    ]
    .sort_values(
        "Correlation with TARGET",
        ascending=True
    )
    .head(10)
)

col1, col2 = st.columns(2)

with col1:
    fig_positive = px.bar(
        positive_corr_df,
        x="Feature",
        y="Correlation with TARGET",
        text="Correlation with TARGET",
        title="Top Positive Correlations"
    )

    fig_positive.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_positive,
        use_container_width=True
    )

with col2:
    fig_negative = px.bar(
        negative_corr_df,
        x="Feature",
        y="Correlation with TARGET",
        text="Correlation with TARGET",
        title="Top Negative Correlations"
    )

    fig_negative.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_negative,
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
# 4. CREDIT VS INCOME SCATTER PLOT
# =====================================================

st.subheader("💳 Credit vs Income Scatter Plot")

fig_credit_income = px.scatter(
    scatter_df,
    x="AMT_INCOME_TOTAL",
    y="AMT_CREDIT",
    color="TARGET",
    title="Credit vs Income Scatter Plot"
)

st.plotly_chart(
    fig_credit_income,
    use_container_width=True
)

st.divider()

# =====================================================
# 5. EXTERNAL SCORE VS TARGET
# =====================================================

st.subheader("📉 External Score vs TARGET")

external_target_df = (
    filtered_df.groupby("TARGET")
    .agg(
        EXT_SOURCE_1=("EXT_SOURCE_1", "mean"),
        EXT_SOURCE_2=("EXT_SOURCE_2", "mean"),
        EXT_SOURCE_3=("EXT_SOURCE_3", "mean"),
        AVG_EXT_SCORE=("AVG_EXT_SCORE", "mean")
    )
    .reset_index()
)

external_target_melt = external_target_df.melt(
    id_vars="TARGET",
    var_name="External Score",
    value_name="Average Score"
)

fig_ext_target = px.bar(
    external_target_melt,
    x="External Score",
    y="Average Score",
    color="TARGET",
    barmode="group",
    title="External Score vs TARGET"
)

st.plotly_chart(
    fig_ext_target,
    use_container_width=True
)

st.divider()

# =====================================================
# IMPORTANT RISK FACTOR ANALYSIS
# =====================================================

st.subheader("⚠ Important Risk Factors")

# Low external score
low_score_df = filtered_df[
    filtered_df["AVG_EXT_SCORE"] < 0.35
]

low_score_default_rate = (
    low_score_df["TARGET"]
    .mean()
) * 100

# High credit to income
high_credit_income_df = filtered_df[
    filtered_df["CREDIT_INCOME_RATIO"] > 6
]

high_credit_income_default_rate = (
    high_credit_income_df["TARGET"]
    .mean()
) * 100

# High annuity to income
high_annuity_income_df = filtered_df[
    filtered_df["ANNUITY_INCOME_RATIO"] > 0.35
]

high_annuity_income_default_rate = (
    high_annuity_income_df["TARGET"]
    .mean()
) * 100

# Younger age
filtered_df["AGE_GROUP"] = pd.cut(
    filtered_df["AGE"],
    bins=[18, 30, 45, 60, 100],
    labels=[
        "18-30",
        "31-45",
        "46-60",
        "61+"
    ]
)

age_risk_df = (
    filtered_df.groupby(
        "AGE_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .reset_index()
)

age_risk_df["Default Rate %"] = (
    age_risk_df["TARGET"] * 100
)

highest_risk_age_group = (
    age_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["AGE_GROUP"]
)

# Regional risk
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

# Employment risk
employment_risk_df = (
    filtered_df.groupby(
        filtered_df["EMPLOYMENT_YEARS"].round()
    )["TARGET"]
    .mean()
    .reset_index()
)

employment_risk_df["Default Rate %"] = (
    employment_risk_df["TARGET"] * 100
)

# Occupation risk
occupation_risk_df = (
    filtered_df.groupby("OCCUPATION_TYPE")["TARGET"]
    .mean()
    .reset_index()
)

occupation_risk_df["Default Rate %"] = (
    occupation_risk_df["TARGET"] * 100
)

highest_risk_occupation = (
    occupation_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["OCCUPATION_TYPE"]
)

# Income type risk
income_type_risk_df = (
    filtered_df.groupby("NAME_INCOME_TYPE")["TARGET"]
    .mean()
    .reset_index()
)

income_type_risk_df["Default Rate %"] = (
    income_type_risk_df["TARGET"] * 100
)

highest_risk_income_type = (
    income_type_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_INCOME_TYPE"]
)

risk_factor_df = pd.DataFrame(
    {
        "Risk Factor": [
            "Low External Credit Score",
            "High Credit-to-Income Ratio",
            "High Annuity-to-Income Ratio",
            "Younger Age Group",
            "Regional Risk Rating",
            "Employment History",
            "Occupation Type",
            "Income Type"
        ],
        "Indicator": [
            "AVG_EXT_SCORE < 0.35",
            "CREDIT_INCOME_RATIO > 6",
            "ANNUITY_INCOME_RATIO > 0.35",
            str(highest_risk_age_group),
            str(highest_risk_region),
            "Employment Years Pattern",
            highest_risk_occupation,
            highest_risk_income_type
        ],
        "Default Rate / Note": [
            f"{low_score_default_rate:.2f}%",
            f"{high_credit_income_default_rate:.2f}%",
            f"{high_annuity_income_default_rate:.2f}%",
            "Highest risk age group",
            "Highest risk region rating",
            "Default varies by employment tenure",
            "Highest risk occupation",
            "Highest risk income type"
        ]
    }
)

st.dataframe(
    risk_factor_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

st.subheader("💡 Correlation & Risk Factor Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Strongest positive correlation with TARGET: **{strongest_positive_feature} ({strongest_positive_value:.3f})**

✅ Strongest negative correlation with TARGET: **{strongest_negative_feature} ({strongest_negative_value:.3f})**

✅ Average External Score: **{avg_external_score:.3f}**

✅ External scores are important indicators for credit-risk behavior.
"""
    )

with col2:
    st.warning(
        f"""
⚠ Low external credit score default rate: **{low_score_default_rate:.2f}%**

⚠ High credit-to-income default rate: **{high_credit_income_default_rate:.2f}%**

⚠ High annuity-to-income default rate: **{high_annuity_income_default_rate:.2f}%**

⚠ Risk factors should be carefully handled before model building.
"""
    )