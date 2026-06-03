import streamlit as st
import pandas as pd
import google.generativeai as genai

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("🤖 AI Data Analyst Assistant")

st.write(
    "Upload any CSV file and perform AI-powered Exploratory Data Analysis."
)

# -----------------------------
# GEMINI SETUP
# -----------------------------
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
except:
    model = None

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # =============================
    # DATA PREVIEW
    # =============================
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =============================
    # DATASET SHAPE
    # =============================
    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # =============================
    # MISSING VALUES
    # =============================
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    # =============================
    # DUPLICATES
    # =============================
    st.subheader("Duplicate Records")
    duplicates = df.duplicated().sum()
    st.write(duplicates)

    # =============================
    # DATA TYPES
    # =============================
    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    # =============================
    # SUMMARY STATS
    # =============================
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # =============================
    # SIMPLE VISUALIZATION
    # =============================
    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numeric_columns) > 0:

        st.subheader("Visualization")

        selected_column = st.selectbox(
            "Choose a numeric column",
            numeric_columns
        )

        chart_data = (
            df[selected_column]
            .value_counts()
            .head(20)
        )

        st.bar_chart(chart_data)

    # =============================
    # AI INSIGHTS
    # =============================
    st.subheader("🧠 AI Insights")

    missing = df.isnull().sum().sum()

    st.write(
        f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    st.write(
        f"Total missing values: {missing}"
    )

    st.write(
        f"Duplicate rows: {duplicates}"
    )

    if missing == 0:
        st.success(
            "Dataset is clean with no missing values."
        )
    else:
        st.warning(
            "Dataset contains missing values and may require cleaning."
        )

    # =============================
    # AI CHATBOT
    # =============================
    st.subheader("💬 Ask AI About Your Dataset")

    user_question = st.text_input(
        "Ask any question about your dataset"
    )

    if user_question:

        if model is not None:

            dataset_context = f"""
            Dataset Shape:
            {df.shape}

            Columns:
            {list(df.columns)}

            Missing Values:
            {df.isnull().sum().to_string()}

            Summary Statistics:
            {df.describe().to_string()}

            First 5 Rows:
            {df.head().to_string()}
            """

            prompt = f"""
            You are a professional Data Analyst.

            Dataset Information:

            {dataset_context}

            User Question:
            {user_question}

            Give a clear data analysis answer.
            """

            with st.spinner("Analyzing..."):

                response = model.generate_content(
                    prompt
                )

                st.success("AI Response")

                st.write(
                    response.text
                )

        else:
            st.error(
                "Gemini API key not configured."
            )

    # =============================
    # DOWNLOAD REPORT
    # =============================
    report = df.describe().to_csv()

    st.download_button(
        label="📥 Download Analysis Report",
        data=report,
        file_name="analysis_report.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Upload a CSV file to begin analysis."
    )