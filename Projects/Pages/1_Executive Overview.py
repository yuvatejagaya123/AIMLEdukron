import streamlit as st
import pandas as pd

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.filters import sidebar_filters
from utils.kpis import calculate_kpis

from utils.charts import (donut_chart,pie_chart,bar_chart,histogram)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Executive Overview",page_icon="📊",layout="wide")


# =====================================================
# PAGE HEADER
# =====================================================

st.title("📊 Executive Overview")

st.markdown(
    """
    This page provides management with an overall picture of loan applicants,
    credit exposure, repayment behavior, and default risk.
    """
)


# =====================================================
# LOAD DATA
# =====================================================

try:
    df = load_data("Data/application_train.csv")

    df = clean_data(df)

    df = create_features(df)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Executive Overview Filters")

df = sidebar_filters(df)

if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# =====================================================
# KPI CALCULATIONS
# =====================================================

metrics = calculate_kpis(df)

total_applications = metrics["total_applications"]
default_customers = metrics["default_customers"]
non_default_customers = metrics["non_default_customers"]
default_rate = metrics["default_rate"]

total_credit_amount = df["AMT_CREDIT"].sum()
average_credit_amount = df["AMT_CREDIT"].mean()
average_income = df["AMT_INCOME_TOTAL"].mean()
average_annuity = df["AMT_ANNUITY"].mean()


# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Applications",f"{total_applications:,}")

col2.metric("Total Default Customers",f"{default_customers:,}")

col3.metric("Total Non-Default Customers",f"{non_default_customers:,}")

col4.metric("Default Rate %",f"{default_rate:.2f}%")

col5, col6, col7, col8 = st.columns(4)

col5.metric("Total Credit Amount",f"{total_credit_amount:,.0f}")

col6.metric("Average Credit Amount",f"{average_credit_amount:,.0f}")

col7.metric("Average Income",f"{average_income:,.0f}")

col8.metric("Average Annuity",f"{average_annuity:,.0f}")

st.divider()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_mode_value(dataframe, column_name):
    if column_name in dataframe.columns and not dataframe[column_name].dropna().empty:
        return dataframe[column_name].mode()[0]
    return "Not Available"


def get_highest_risk_segment(dataframe, column_name):
    if column_name in dataframe.columns:
        risk_table = (dataframe.groupby(column_name)["TARGET"].mean().dropna())

        if not risk_table.empty:
            segment_name = risk_table.idxmax()
            segment_rate = risk_table.max() * 100
            return segment_name, segment_rate

    return "Not Available", 0


# =====================================================
# IMPORTANT INSIGHT CALCULATIONS
# =====================================================

most_common_income_type = get_mode_value(df,"NAME_INCOME_TYPE")

most_common_education_level = get_mode_value(df,"NAME_EDUCATION_TYPE")

highest_risk_segment, highest_risk_segment_rate = get_highest_risk_segment(df,"NAME_INCOME_TYPE")


# =====================================================
# IMPORTANT INSIGHTS
# =====================================================

st.subheader("💡 Important Insights")

col1, col2, col3 = st.columns(3)

col1.info(
    f"""
### Overall Default Rate

**{default_rate:.2f}%**

This represents the percentage of applicants who had payment difficulties.
"""
)

col2.info(
    f"""
### Average Customer Income

**{average_income:,.0f}**

This shows the average income level of applicants in the selected data.
"""
)

col3.info(
    f"""
### Average Loan Amount

**{average_credit_amount:,.0f}**

This shows the average credit amount requested by applicants.
"""
)

col4, col5, col6 = st.columns(3)

col4.success(
    f"""
### Most Common Income Type

**{most_common_income_type}**
"""
)

col5.success(
    f"""
### Most Common Education Level

**{most_common_education_level}**
"""
)

col6.warning(
    f"""
### Highest Risk Customer Segment

**{highest_risk_segment}**

Default Rate: **{highest_risk_segment_rate:.2f}%**
"""
)

st.divider()


# =====================================================
# VISUALIZATION 1
# DEFAULT VS NON DEFAULT CUSTOMERS
# =====================================================

st.subheader("🎯 Default vs Non-Default Customers")

st.plotly_chart(donut_chart(df,"TARGET","Default vs Non-Default Customers"),use_container_width=True)

st.divider()


# =====================================================
# VISUALIZATIONS 2 TO 4
# APPLICATION BREAKDOWN
# =====================================================

st.subheader("👥 Applicant Breakdown")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(pie_chart(df,"CODE_GENDER","Total Applications by Gender"),use_container_width=True)

with col2:
    st.plotly_chart(donut_chart(df,"NAME_CONTRACT_TYPE","Applications by Contract Type"),use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(bar_chart(df,"NAME_INCOME_TYPE",None,"Applications by Income Type"),use_container_width=True)

with col2:
    st.plotly_chart(bar_chart(df,"NAME_EDUCATION_TYPE",None,"Applications by Education Level"),use_container_width=True)

st.divider()


# =====================================================
# VISUALIZATION 5
# CREDIT AMOUNT DISTRIBUTION
# =====================================================

st.subheader("💳 Credit Amount Distribution")

st.plotly_chart(histogram(df,"AMT_CREDIT","Credit Amount Distribution"),use_container_width=True)

st.divider()


# =====================================================
# VISUALIZATION 6
# MONTHLY / OVERALL APPLICANT SUMMARY
# =====================================================

st.subheader("📋 Overall Applicant Summary")

summary_df = pd.DataFrame(
    {
        "Metric": [
            "Total Applications",
            "Default Customers",
            "Non-Default Customers",
            "Default Rate %",
            "Total Credit Amount",
            "Average Credit Amount",
            "Average Income",
            "Average Annuity"
        ],
        "Value": [
            f"{total_applications:,}",
            f"{default_customers:,}",
            f"{non_default_customers:,}",
            f"{default_rate:.2f}%",
            f"{total_credit_amount:,.0f}",
            f"{average_credit_amount:,.0f}",
            f"{average_income:,.0f}",
            f"{average_annuity:,.0f}"
        ]
    }
)

st.dataframe(summary_df,use_container_width=True,hide_index=True)


# =====================================================
# APPLICATION SUMMARY BY SEGMENTS
# =====================================================

st.subheader("📊 Applicant Summary by Segment")

col1, col2 = st.columns(2)

with col1:
    income_summary = (
        df.groupby("NAME_INCOME_TYPE")
        .agg(
            Applications=("SK_ID_CURR", "count"),
            Default_Rate=("TARGET", "mean"),
            Average_Credit=("AMT_CREDIT", "mean"),
            Average_Income=("AMT_INCOME_TOTAL", "mean")
        )
        .reset_index()
    )

    income_summary["Default_Rate"] = income_summary["Default_Rate"] * 100

    st.write("#### Income Type Summary")

    st.dataframe(income_summary,use_container_width=True,hide_index=True)

with col2:
    contract_summary = (
        df.groupby("NAME_CONTRACT_TYPE")
        .agg(
            Applications=("SK_ID_CURR", "count"),
            Default_Rate=("TARGET", "mean"),
            Average_Credit=("AMT_CREDIT", "mean"),
            Average_Annuity=("AMT_ANNUITY", "mean")
        )
        .reset_index()
    )

    contract_summary["Default_Rate"] = contract_summary["Default_Rate"] * 100

    st.write("#### Contract Type Summary")

    st.dataframe(contract_summary,use_container_width=True,hide_index=True)

st.divider()


# =====================================================
# FILTERED DATA PREVIEW
# =====================================================

with st.expander("🔍 View Filtered Applicant Data"):

    display_columns = [
        "SK_ID_CURR",
        "TARGET",
        "CODE_GENDER",
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE"
    ]

    available_columns = [
        col for col in display_columns
        if col in df.columns
    ]

    st.dataframe(
        df[available_columns].head(100),
        use_container_width=True
    )