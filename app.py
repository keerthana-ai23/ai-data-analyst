import streamlit as st
import pandas as pd

st.set_page_config(
page_title="GenAI Data Analyst Assistant",
layout="wide"
)

st.title("🤖 GenAI-Powered Data Analyst Assistant")

st.write(
"Upload any CSV dataset and receive automated analysis, insights, recommendations, and AI-style explanations."
)

uploaded_file = st.file_uploader(
"Upload CSV File",
type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

else:
    st.info(
        "Upload a CSV file to begin analysis."
    )


# ====================================
# DATASET HEALTH SCORE
# ====================================

st.header("📊 Dataset Health Score")

score = 100

if missing > 0:
    score -= 20

if duplicates > 0:
    score -= 15

if len(numeric_df.columns) == 0:
    score -= 25

st.metric(
    "Health Score",
    f"{score}/100"
)

# ====================================
# DATASET OVERVIEW
# ====================================

st.header("📄 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Numeric Features",
        len(numeric_df.columns)
    )

st.dataframe(
    df.head()
)

# ====================================
# AI EXECUTIVE SUMMARY
# ====================================

st.header("🧠 AI Executive Summary")

summary = f"""
This dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

The dataset contains {missing} missing values and {duplicates} duplicate records.

There are {len(numeric_df.columns)} numeric variables available for statistical analysis.

The dataset appears suitable for exploratory data analysis, machine learning, predictive analytics, and business intelligence reporting.
"""

st.info(summary)

# ====================================
# DATA QUALITY
# ====================================

st.header("🧹 Data Quality Assessment")

if missing == 0:
    st.success("No missing values detected.")
else:
    st.warning(
        f"{missing} missing values detected."
    )

if duplicates == 0:
    st.success("No duplicate rows detected.")
else:
    st.warning(
        f"{duplicates} duplicate rows detected."
    )

# ====================================
# TOP CORRELATIONS
# ====================================

if len(numeric_df.columns) > 1:

    st.header("📈 Top Correlations")

    corr_matrix = numeric_df.corr()

    corr_pairs = (
        corr_matrix.abs()
        .unstack()
        .sort_values(
            ascending=False
        )
    )

    corr_pairs = corr_pairs[
        corr_pairs < 1
    ]

    shown = set()

    for pair, value in corr_pairs.items():

        if pair not in shown:

            st.write(
                f"{pair[0]} ↔ {pair[1]} = {value:.2f}"
            )

            shown.add(pair)

            if len(shown) == 5:
                break

# ====================================
# AI RECOMMENDATIONS
# ====================================

st.header("💡 AI Recommendations")

recommendations = []

if duplicates > 0:
    recommendations.append(
        "Remove duplicate rows before modeling."
    )

if missing > 0:
    recommendations.append(
        "Handle missing values using imputation or removal."
    )

recommendations.append(
    "Investigate highly correlated features."
)

recommendations.append(
    "Consider building predictive models."
)

recommendations.append(
    "Perform feature engineering."
)

for rec in recommendations:
    st.write(f"• {rec}")

# ====================================
# DATASET CHAT
# ====================================

st.header("💬 Ask About Your Dataset")

question = st.text_input(
    "Ask a question"
)

if question:

    q = question.lower()

    if "row" in q:
        st.success(
            f"The dataset contains {df.shape[0]} rows."
        )

    elif "column" in q:
        st.success(
            f"The dataset contains {df.shape[1]} columns."
        )

    elif "missing" in q:
        st.success(
            f"The dataset contains {missing} missing values."
        )

    elif "duplicate" in q:
        st.success(
            f"The dataset contains {duplicates} duplicate rows."
        )

    elif "important" in q or "correlation" in q:

        if len(numeric_df.columns) > 1:

            strongest = corr_pairs.index[0]

            st.success(
                f"The strongest relationship is between {strongest[0]} and {strongest[1]}."
            )

    elif "machine learning" in q:

        st.success(
            "Yes. This dataset is suitable for machine learning if the target variable is clearly defined."
        )

    elif "tell" in q or "summary" in q:

        st.success(summary)

    else:

        st.info(
            "Try asking about rows, columns, missing values, duplicates, correlations, machine learning, or summary."
        )

# ====================================
# DOWNLOAD REPORT
# ====================================

report = df.describe(
    include="all"
).to_csv()

st.download_button(
    label="📥 Download Report",
    data=report,
    file_name="analysis_report.csv",
    mime="text/csv"
)

