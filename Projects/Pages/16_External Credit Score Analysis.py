import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.charts import histogram

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="External Credit Score Analysis",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📊 External Credit Score Analysis")

st.markdown("""
Analyze external credit score indicators and
their relationship with default behavior.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

df = clean_data(df)

df = create_features(df)

# =====================================================
# AVERAGE EXTERNAL SCORE
# =====================================================

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

st.sidebar.header("External Score Filters")

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    df["TARGET"].isin(target_filter)
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

avg_ext1 = (
    filtered_df["EXT_SOURCE_1"]
    .mean()
)

avg_ext2 = (
    filtered_df["EXT_SOURCE_2"]
    .mean()
)

avg_ext3 = (
    filtered_df["EXT_SOURCE_3"]
    .mean()
)

missing_ext_records = (
    filtered_df[
        [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        ]
    ]
    .isna()
    .any(axis=1)
    .sum()
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average EXT_SOURCE_1",
    f"{avg_ext1:.3f}"
)

col2.metric(
    "Average EXT_SOURCE_2",
    f"{avg_ext2:.3f}"
)

col3.metric(
    "Average EXT_SOURCE_3",
    f"{avg_ext3:.3f}"
)

col4.metric(
    "Missing External Score Records",
    f"{missing_ext_records:,}"
)

st.divider()

# =====================================================
# EXT_SOURCE_1 DISTRIBUTION
# =====================================================

st.subheader("EXT_SOURCE_1 Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "EXT_SOURCE_1",
        "EXT_SOURCE_1 Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# EXT_SOURCE_2 DISTRIBUTION
# =====================================================

st.subheader("EXT_SOURCE_2 Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "EXT_SOURCE_2",
        "EXT_SOURCE_2 Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# EXT_SOURCE_3 DISTRIBUTION
# =====================================================

st.subheader("EXT_SOURCE_3 Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "EXT_SOURCE_3",
        "EXT_SOURCE_3 Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# EXTERNAL SCORES BY TARGET
# =====================================================

target_score_df = (
    filtered_df.groupby("TARGET")
    .agg(
        EXT_SOURCE_1=("EXT_SOURCE_1", "mean"),
        EXT_SOURCE_2=("EXT_SOURCE_2", "mean"),
        EXT_SOURCE_3=("EXT_SOURCE_3", "mean")
    )
    .reset_index()
)

target_score_melt = target_score_df.melt(
    id_vars="TARGET",
    var_name="External Score",
    value_name="Average Score"
)

fig_target_scores = px.bar(
    target_score_melt,
    x="External Score",
    y="Average Score",
    color="TARGET",
    barmode="group",
    title="External Scores by TARGET"
)

st.plotly_chart(
    fig_target_scores,
    use_container_width=True
)

st.divider()

# =====================================================
# SAMPLE FOR SCATTER PLOTS
# =====================================================

scatter_df = filtered_df.dropna(
    subset=[
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]
)

scatter_df = scatter_df.sample(
    min(10000, len(scatter_df)),
    random_state=42
)

# =====================================================
# EXT_SOURCE_1 vs EXT_SOURCE_2
# =====================================================

fig_ext12 = px.scatter(
    scatter_df,
    x="EXT_SOURCE_1",
    y="EXT_SOURCE_2",
    color="TARGET",
    title="EXT_SOURCE_1 vs EXT_SOURCE_2"
)

st.plotly_chart(
    fig_ext12,
    use_container_width=True
)

st.divider()

# =====================================================
# EXT_SOURCE_2 vs EXT_SOURCE_3
# =====================================================

fig_ext23 = px.scatter(
    scatter_df,
    x="EXT_SOURCE_2",
    y="EXT_SOURCE_3",
    color="TARGET",
    title="EXT_SOURCE_2 vs EXT_SOURCE_3"
)

st.plotly_chart(
    fig_ext23,
    use_container_width=True
)

st.divider()

# =====================================================
# EXTERNAL SCORE GROUPS
# =====================================================

df["SCORE_GROUP"] = pd.cut(
    df["AVG_EXT_SCORE"],
    bins=[0, 0.25, 0.50, 0.75, 1.00],
    labels=[
        "Low Score",
        "Medium Score",
        "High Score",
        "Very High Score"
    ]
)

score_risk_df = (
    df.groupby(
        "SCORE_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .reset_index()
)

score_risk_df["Default Rate %"] = (
    score_risk_df["TARGET"] * 100
)

# =====================================================
# EXTERNAL SCORE vs DEFAULT RATE
# =====================================================

fig_score_risk = px.bar(
    score_risk_df,
    x="SCORE_GROUP",
    y="Default Rate %",
    text="Default Rate %",
    title="External Score vs Default Rate"
)

fig_score_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_score_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# HIGH SCORE vs LOW SCORE ANALYSIS
# =====================================================

high_score_default_rate = (
    df[
        df["SCORE_GROUP"]
        == "Very High Score"
    ]["TARGET"]
    .mean()
) * 100

low_score_default_rate = (
    df[
        df["SCORE_GROUP"]
        == "Low Score"
    ]["TARGET"]
    .mean()
) * 100

# =====================================================
# SUMMARY TABLE
# =====================================================

st.subheader("📋 External Score Summary")

score_summary = (
    df.groupby(
        "SCORE_GROUP",
        observed=True
    )
    .agg(
        Customers=("SK_ID_CURR", "count"),
        Avg_EXT1=("EXT_SOURCE_1", "mean"),
        Avg_EXT2=("EXT_SOURCE_2", "mean"),
        Avg_EXT3=("EXT_SOURCE_3", "mean"),
        Default_Rate=("TARGET", "mean")
    )
    .reset_index()
)

score_summary["Default_Rate"] = (
    score_summary["Default_Rate"] * 100
).round(2)

st.dataframe(
    score_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

highest_risk_score_group = (
    score_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["SCORE_GROUP"]
)

lowest_risk_score_group = (
    score_risk_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["SCORE_GROUP"]
)

st.subheader("💡 External Credit Score Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Average EXT_SOURCE_1: {avg_ext1:.3f}

✅ Average EXT_SOURCE_2: {avg_ext2:.3f}

✅ Average EXT_SOURCE_3: {avg_ext3:.3f}

✅ Lowest Risk Score Group: {lowest_risk_score_group}

✅ High external scores generally indicate lower credit risk.
"""
    )

with col2:
    st.warning(
        f"""
⚠ Missing External Score Records: {missing_ext_records:,}

⚠ Highest Risk Score Group: {highest_risk_score_group}

⚠ Low Score Default Rate: {low_score_default_rate:.2f}%

⚠ Very High Score Default Rate: {high_score_default_rate:.2f}%

⚠ External scores are among the strongest indicators of default behavior.
"""
    )