import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.filters import sidebar_filters
from utils.risk_scoring import add_risk_columns

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Risk Scoring Center",
    page_icon="🚨",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🚨 Risk Scoring Center")

st.markdown("""
Customer risk segmentation using a rule-based
credit risk scoring framework.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data(
    "Data/application_train.csv"
)

df = clean_data(df)

df = create_features(df)

# =====================================================
# ADD AGE
# =====================================================

df["AGE"] = (
    abs(df["DAYS_BIRTH"]) / 365
)

# =====================================================
# ADD RISK COLUMNS
# =====================================================

df = add_risk_columns(df)

# =====================================================
# FILTERS
# =====================================================

filtered_df = sidebar_filters(df)

# =====================================================
# KPI CARDS
# =====================================================

low_risk = (
    filtered_df["RISK_LEVEL"]
    == "Low Risk"
).sum()

medium_risk = (
    filtered_df["RISK_LEVEL"]
    == "Medium Risk"
).sum()

high_risk = (
    filtered_df["RISK_LEVEL"]
    == "High Risk"
).sum()

critical_risk = (
    filtered_df["RISK_LEVEL"]
    == "Critical Risk"
).sum()

st.subheader("📌 Risk Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Low Risk",
    f"{low_risk:,}"
)

col2.metric(
    "Medium Risk",
    f"{medium_risk:,}"
)

col3.metric(
    "High Risk",
    f"{high_risk:,}"
)

col4.metric(
    "Critical Risk",
    f"{critical_risk:,}"
)

st.divider()

# =====================================================
# RISK DISTRIBUTION
# =====================================================

risk_count_df = (
    filtered_df["RISK_LEVEL"]
    .value_counts()
    .reset_index()
)

risk_count_df.columns = [
    "Risk Level",
    "Customers"
]

fig_risk_dist = px.bar(
    risk_count_df,
    x="Risk Level",
    y="Customers",
    text="Customers",
    color="Risk Level",
    title="Customer Risk Distribution"
)

st.plotly_chart(
    fig_risk_dist,
    use_container_width=True
)

st.divider()

# =====================================================
# AVERAGE RISK SCORE
# =====================================================

avg_risk_score = (
    filtered_df["RISK_SCORE"]
    .mean()
)

st.metric(
    "Average Risk Score",
    f"{avg_risk_score:.1f}"
)

st.divider()

# =====================================================
# RISK GAUGE
# =====================================================

st.subheader("🎯 Portfolio Risk Gauge")

fig_gauge = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=avg_risk_score,

        title={
            "text":"Average Portfolio Risk Score"
        },

        gauge={

            "axis":{
                "range":[0,100]
            },

            "bar":{
                "color":"darkred"
            },

            "steps":[

                {
                    "range":[0,25],
                    "color":"green"
                },

                {
                    "range":[25,50],
                    "color":"yellow"
                },

                {
                    "range":[50,75],
                    "color":"orange"
                },

                {
                    "range":[75,100],
                    "color":"red"
                }
            ]
        }
    )
)

st.plotly_chart(
    fig_gauge,
    use_container_width=True
)

st.divider()



# =====================================================
# RISK SCORE DISTRIBUTION
# =====================================================

fig_score_dist = px.histogram(
    filtered_df,
    x="RISK_SCORE",
    nbins=30,
    title="Risk Score Distribution"
)

st.plotly_chart(
    fig_score_dist,
    use_container_width=True
)

st.divider()

# =====================================================
# RISK BY GENDER
# =====================================================

gender_risk_df = (
    filtered_df.groupby(
        "CODE_GENDER"
    )["RISK_SCORE"]
    .mean()
    .reset_index()
)

fig_gender = px.bar(
    gender_risk_df,
    x="CODE_GENDER",
    y="RISK_SCORE",
    title="Average Risk Score by Gender"
)

st.plotly_chart(
    fig_gender,
    use_container_width=True
)

st.divider()

# =====================================================
# RISK BY INCOME TYPE
# =====================================================

income_risk_df = (
    filtered_df.groupby(
        "NAME_INCOME_TYPE"
    )["RISK_SCORE"]
    .mean()
    .reset_index()
)

fig_income = px.bar(
    income_risk_df,
    x="NAME_INCOME_TYPE",
    y="RISK_SCORE",
    title="Average Risk Score by Income Type"
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)

st.divider()

# =====================================================
# RISK BY EDUCATION
# =====================================================

education_risk_df = (
    filtered_df.groupby(
        "NAME_EDUCATION_TYPE"
    )["RISK_SCORE"]
    .mean()
    .reset_index()
)

fig_education = px.bar(
    education_risk_df,
    x="NAME_EDUCATION_TYPE",
    y="RISK_SCORE",
    title="Average Risk Score by Education"
)

st.plotly_chart(
    fig_education,
    use_container_width=True
)

st.divider()

# =====================================================
# RISK SCORE VS EXTERNAL SCORE
# =====================================================

sample_df = filtered_df.sample(
    min(10000, len(filtered_df)),
    random_state=42
)

fig_ext = px.scatter(
    sample_df,
    x="AVG_EXT_SCORE",
    y="RISK_SCORE",
    color="RISK_LEVEL",
    title="Risk Score vs External Score"
)

st.plotly_chart(
    fig_ext,
    use_container_width=True
)

st.divider()

# =====================================================
# RISK SCORE VS CREDIT INCOME
# =====================================================

fig_ci = px.scatter(
    sample_df,
    x="CREDIT_INCOME_RATIO",
    y="RISK_SCORE",
    color="RISK_LEVEL",
    title="Risk Score vs Credit-To-Income Ratio"
)

st.plotly_chart(
    fig_ci,
    use_container_width=True
)

st.divider()

# =====================================================
# HIGH RISK CUSTOMERS
# =====================================================

st.subheader("🚨 High Risk Customers")

high_risk_df = filtered_df[
    filtered_df["RISK_LEVEL"].isin(
        [
            "High Risk",
            "Critical Risk"
        ]
    )
]

display_cols = [
    "SK_ID_CURR",
    "TARGET",
    "RISK_SCORE",
    "RISK_LEVEL",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AVG_EXT_SCORE"
]

st.dataframe(
    high_risk_df[display_cols],
    use_container_width=True,
    hide_index=True
)

st.divider()


st.download_button(
    "⬇ Download High Risk Customers",
    high_risk_df.to_csv(index=False),
    "high_risk_customers.csv",
    "text/csv"
)




# =====================================================
# INSIGHTS
# =====================================================

highest_risk_group = (
    risk_count_df
    .sort_values(
        "Customers",
        ascending=False
    )
    .iloc[0]["Risk Level"]
)

st.subheader("💡 Risk Scoring Insights")

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
✅ Average Risk Score: {avg_risk_score:.1f}

✅ Largest Segment:
{highest_risk_group}

✅ Risk scoring enables customer segmentation.
"""
    )

with col2:

    st.warning(
        f"""
⚠ High Risk Customers:
{len(high_risk_df):,}

⚠ Combine risk score with external scores
for better decision making.
"""
    )
