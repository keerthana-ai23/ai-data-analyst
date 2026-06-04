import streamlit as st
import pandas as pd
import requests

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="GenAI Data Analyst Assistant",
    layout="wide"
)

st.title("🤖 GenAI-Powered Data Analyst Assistant")

st.write(
    "Upload any CSV dataset and receive automated analysis, insights, recommendations, and AI-style explanations."
)

# ==========================
# HUGGING FACE SETUP
# ==========================

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

API_URL = (
    "https://api-inference.huggingface.co/models/"
    "mistralai/Mistral-7B-Instruct-v0.2"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def ask_llm(prompt):

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300
            }
        }
    )

    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]

    return str(result)


# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

missing = df.isnull().sum().sum()
duplicates = df.duplicated().sum()

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

# ==========================
# DATASET HEALTH SCORE
# ==========================

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

# ==========================
# DATASET OVERVIEW
# ==========================

st.header("📄 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric(
        "Numeric Features",
        len(numeric_df.columns)
    )

st.dataframe(df.head())

# ==========================
# EXECUTIVE SUMMARY
# ==========================

st.header("🧠 Executive Summary")

summary = f"""
Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

Missing Values: {missing}

Duplicate Records: {duplicates}

Numeric Features: {len(numeric_df.columns)}

This dataset appears suitable for analytics, reporting, machine learning, and predictive modeling.
"""

st.info(summary)

# ==========================
# DATA QUALITY
# ==========================

st.header("🧹 Data Quality Assessment")

if missing == 0:
    st.success("No missing values detected.")
else:
    st.warning(f"{missing} missing values detected.")

if duplicates == 0:
    st.success("No duplicate rows detected.")
else:
    st.warning(f"{duplicates} duplicate rows detected.")

# ==========================
# CORRELATIONS
# ==========================

strongest_pair = None

if len(numeric_df.columns) > 1:

    st.header("📈 Top Correlations")

    corr_matrix = numeric_df.corr()

    corr_pairs = (
        corr_matrix.abs()
        .unstack()
        .sort_values(ascending=False)
    )

    corr_pairs = corr_pairs[corr_pairs < 1]

    displayed = 0

    for pair, value in corr_pairs.items():

        st.write(
            f"{pair[0]} ↔ {pair[1]} = {value:.2f}"
        )

        if strongest_pair is None:
            strongest_pair = pair

        displayed += 1

        if displayed == 5:
            break

# ==========================
# AI RECOMMENDATIONS
# ==========================

st.header("💡 AI Recommendations")

if duplicates > 0:
    st.write("• Remove duplicate rows before modeling.")

if missing > 0:
    st.write("• Handle missing values.")

st.write("• Investigate highly correlated features.")
st.write("• Perform feature engineering.")
st.write("• Consider predictive modeling.")
st.write("• Evaluate business KPIs and trends.")

# ==========================
# DATASET CHAT
# ==========================

st.header("💬 Ask About Your Dataset")

question = st.text_input(
    "Ask a question"
)

if question:

    q = question.lower()

    if "how many rows" in q:
        st.success(
            f"The dataset contains {df.shape[0]} rows."
        )

    elif "how many columns" in q:
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

    elif (
        "strongest" in q
        or "correlation" in q
        or "relationship" in q
    ):

        if strongest_pair:

            st.success(
                f"The strongest relationship is between {strongest_pair[0]} and {strongest_pair[1]}."
            )

    elif "summary" in q:
        st.success(summary)

    else:
        st.info(
            "Use GenAI Analysis below for deeper questions."
        )

# ==========================
# GENAI ANALYSIS
# ==========================

dataset_context = f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{list(df.columns)}

Missing Values:
{missing}

Duplicate Records:
{duplicates}
"""

prompt = f"""
You are a Senior Data Analyst.

Analyze this dataset:

{dataset_context}

Provide:

1. Executive Summary
2. Key Insights
3. Risks
4. Recommended ML Models
5. Business Recommendations
"""

st.header("🤖 GenAI Analysis")

if st.button("Generate AI Insights"):

    if HF_TOKEN == "":
        st.error(
            "HF_TOKEN not found in Streamlit Secrets."
        )

    else:

        with st.spinner(
            "Generating AI insights..."
        ):

            try:

                result = ask_llm(prompt)

                st.write(result)

            except Exception as e:

                st.error(str(e))

# ==========================
# DOWNLOAD REPORT
# ==========================

report = df.describe(
    include="all"
).to_csv()

st.download_button(
    label="📥 Download Report",
    data=report,
    file_name="analysis_report.csv",
    mime="text/csv"
)