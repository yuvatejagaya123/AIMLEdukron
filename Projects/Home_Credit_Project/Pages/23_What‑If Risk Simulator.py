import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.risk_scoring import (
    calculate_risk_score,
    assign_risk_level
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="What-If Risk Simulator",
    page_icon="🎛️",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🎛️ What-If Risk Simulator")

st.markdown("""
Simulate how changes in customer income, credit amount, annuity,
employment history, age and external credit score affect customer risk.

This page helps business users understand how customer risk changes
under different financial scenarios.
""")

st.divider()

# =====================================================
# INPUT SECTION
# =====================================================

st.subheader("🧾 Customer Scenario Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.slider(
        "Customer Income",
        min_value=50000,
        max_value=1000000,
        value=150000,
        step=10000
    )

with col2:
    credit_amount = st.slider(
        "Credit Amount",
        min_value=50000,
        max_value=2000000,
        value=500000,
        step=10000
    )

with col3:
    annuity = st.slider(
        "Loan Annuity",
        min_value=5000,
        max_value=200000,
        value=30000,
        step=1000
    )

col4, col5, col6 = st.columns(3)

with col4:
    age = st.slider(
        "Customer Age",
        min_value=18,
        max_value=70,
        value=35,
        step=1
    )

with col5:
    employment_years = st.slider(
        "Employment Years",
        min_value=0.0,
        max_value=40.0,
        value=5.0,
        step=0.5
    )

with col6:
    avg_external_score = st.slider(
        "Average External Score",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

st.divider()

# =====================================================
# CALCULATED RATIOS
# =====================================================

credit_income_ratio = (
    credit_amount / income
    if income > 0
    else 0
)

annuity_income_ratio = (
    annuity / income
    if income > 0
    else 0
)

credit_goods_ratio = 1.0

scenario_row = pd.Series(
    {
        "AVG_EXT_SCORE": avg_external_score,
        "CREDIT_INCOME_RATIO": credit_income_ratio,
        "ANNUITY_INCOME_RATIO": annuity_income_ratio,
        "EMPLOYMENT_YEARS": employment_years,
        "AGE": age
    }
)

risk_score = calculate_risk_score(
    scenario_row
)

risk_level = assign_risk_level(
    risk_score
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 Simulated Risk Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Risk Score",
    f"{risk_score}/100"
)

col2.metric(
    "Risk Level",
    risk_level
)

col3.metric(
    "Credit-To-Income Ratio",
    f"{credit_income_ratio:.2f}"
)

col4.metric(
    "Annuity-To-Income Ratio",
    f"{annuity_income_ratio:.3f}"
)

st.divider()

# =====================================================
# RISK GAUGE
# =====================================================

st.subheader("🎯 Risk Score Gauge")

if risk_level == "Low Risk":
    gauge_color = "green"

elif risk_level == "Medium Risk":
    gauge_color = "gold"

elif risk_level == "High Risk":
    gauge_color = "orange"

else:
    gauge_color = "red"

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={
            "text": "Simulated Customer Risk Score"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "bar": {
                "color": gauge_color
            },
            "steps": [
                {
                    "range": [0, 25],
                    "color": "#d4edda"
                },
                {
                    "range": [25, 50],
                    "color": "#fff3cd"
                },
                {
                    "range": [50, 75],
                    "color": "#ffe0b2"
                },
                {
                    "range": [75, 100],
                    "color": "#f8d7da"
                }
            ],
            "threshold": {
                "line": {
                    "color": "black",
                    "width": 4
                },
                "thickness": 0.75,
                "value": risk_score
            }
        }
    )
)

st.plotly_chart(
    fig_gauge,
    use_container_width=True
)

st.divider()