import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from utils.preprocessing import clean_data

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Missing Value Analysis",
    page_icon="🧹",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🧹 Missing Value Analysis")

st.markdown("""
Analyze data quality and identify missing values
before machine learning modeling.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

# =====================================================
# MISSING VALUE CALCULATIONS
# =====================================================

total_rows = df.shape[0]

total_columns = df.shape[1]

total_missing_values = (
    df.isnull()
    .sum()
    .sum()
)

columns_with_missing = (
    df.isnull()
    .sum()
    .gt(0)
    .sum()
)

missing_percent = (
    df.isnull()
    .sum()
    / len(df)
    * 100
)

columns_above_50 = (
    missing_percent > 50
).sum()

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Rows",
    f"{total_rows:,}"
)

col2.metric(
    "Total Columns",
    f"{total_columns:,}"
)

col3.metric(
    "Total Missing Values",
    f"{total_missing_values:,}"
)

col4.metric(
    "Columns with Missing Values",
    f"{columns_with_missing:,}"
)

col5.metric(
    "Columns >50% Missing",
    f"{columns_above_50:,}"
)

st.divider()

# =====================================================
# MISSING VALUE TABLE
# =====================================================

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isnull().sum().values,
    "Missing %": (
        df.isnull().sum() / len(df) * 100
    ).values,
    "Data Type": df.dtypes.astype(str).values
})

missing_df = (
    missing_df
    .sort_values(
        "Missing %",
        ascending=False
    )
)

# =====================================================
# TOP 20 COLUMNS WITH MISSING VALUES
# =====================================================

top20_missing = (
    missing_df
    .head(20)
)

fig_missing = px.bar(
    top20_missing,
    x="Column",
    y="Missing Count",
    text="Missing Count",
    title="Top 20 Columns with Missing Values"
)

st.plotly_chart(
    fig_missing,
    use_container_width=True
)

st.divider()

# =====================================================
# MISSING PERCENTAGE BY COLUMN
# =====================================================

fig_missing_pct = px.bar(
    top20_missing,
    x="Column",
    y="Missing %",
    text="Missing %",
    title="Missing Percentage by Column"
)

fig_missing_pct.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_missing_pct,
    use_container_width=True
)

st.divider()

# =====================================================
# MISSING VALUES HEATMAP
# =====================================================

st.subheader("🔥 Missing Values Heatmap")

heatmap_columns = (
    missing_df[
        missing_df["Missing Count"] > 0
    ]["Column"]
    .head(20)
    .tolist()
)

heatmap_df = df[
    heatmap_columns
].isnull().astype(int)

fig_heatmap = px.imshow(
    heatmap_df.T,
    aspect="auto",
    color_continuous_scale="Reds",
    title="Missing Values Heatmap"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)

st.divider()

# =====================================================
# MISSING VALUES BY DATA TYPE
# =====================================================

dtype_missing = []

for dtype in df.dtypes.astype(str).unique():

    cols = df.select_dtypes(
        include=[dtype]
    ).columns

    missing_count = (
        df[cols]
        .isnull()
        .sum()
        .sum()
    )

    dtype_missing.append(
        {
            "Data Type": dtype,
            "Missing Values": missing_count
        }
    )

dtype_missing_df = pd.DataFrame(
    dtype_missing
)

fig_dtype = px.bar(
    dtype_missing_df,
    x="Data Type",
    y="Missing Values",
    text="Missing Values",
    title="Missing Values by Data Type"
)

st.plotly_chart(
    fig_dtype,
    use_container_width=True
)

st.divider()

# =====================================================
# DATA CLEANING RECOMMENDATIONS
# =====================================================

recommendations = []

for _, row in missing_df.iterrows():

    column = row["Column"]
    missing_pct = row["Missing %"]
    dtype = row["Data Type"]

    if missing_pct > 60:
        action = "Drop"

    elif "float" in dtype:

        if missing_pct > 20:
            action = "Median"

        else:
            action = "Mean"

    elif "int" in dtype:

        action = "Median"

    else:

        if missing_pct > 20:
            action = "Unknown"

        else:
            action = "Mode"

    recommendations.append(
        {
            "Column": column,
            "Missing Count": row["Missing Count"],
            "Missing %": round(
                missing_pct,
                2
            ),
            "Data Type": dtype,
            "Recommended Action": action
        }
    )

recommendation_df = pd.DataFrame(
    recommendations
)

st.subheader("📋 Missing Value Recommendation Table")

st.dataframe(
    recommendation_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# HIGH MISSING COLUMNS
# =====================================================

high_missing_columns = (
    recommendation_df[
        recommendation_df["Missing %"] > 50
    ]
)

st.subheader("⚠ Columns With >50% Missing Data")

st.dataframe(
    high_missing_columns,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

highest_missing_column = (
    missing_df
    .iloc[0]["Column"]
)

highest_missing_pct = (
    missing_df
    .iloc[0]["Missing %"]
)

st.subheader("💡 Missing Value Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Total Rows: {total_rows:,}

✅ Total Columns: {total_columns:,}

✅ Columns with Missing Values: {columns_with_missing:,}

✅ Columns with >50% Missing Data: {columns_above_50:,}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Total Missing Values: {total_missing_values:,}

⚠ Highest Missing Column: {highest_missing_column}

⚠ Highest Missing Percentage: {highest_missing_pct:.2f}%

⚠ Data quality issues should be addressed before model training.
"""
    )
   