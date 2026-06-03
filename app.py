import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("🤖 AI Data Analyst Dashboard")

st.write(
    "Upload any CSV file and automatically perform Exploratory Data Analysis (EDA)."
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Shape
    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # Missing Values
    st.subheader("Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing_df)

    total_missing = df.isnull().sum().sum()

    # Duplicate Records
    st.subheader("Duplicate Records")

    duplicates = df.duplicated().sum()

    st.write(f"Duplicate Rows: {duplicates}")

    # Data Types
    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # AI Insights
    st.subheader("AI Insights")

    st.write(
        f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    st.write(
        f"Total missing values: {total_missing}"
    )

    st.write(
        f"Duplicate rows: {duplicates}"
    )

    # Data Cleaning Suggestions
    st.subheader("Data Cleaning Suggestions")

    if total_missing > 0:
        st.warning(
            f"The dataset contains {total_missing} missing values. Consider filling or removing them."
        )
    else:
        st.success(
            "No missing values detected."
        )

    if duplicates > 0:
        st.warning(
            f"The dataset contains {duplicates} duplicate rows. Consider removing duplicates."
        )
    else:
        st.success(
            "No duplicate rows detected."
        )

    # Column Distribution
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:

        st.subheader("Column Distribution")

        selected_col = st.selectbox(
            "Select a numeric column",
            numeric_cols
        )

        chart_data = (
            df[selected_col]
            .value_counts()
            .head(20)
        )

        st.bar_chart(chart_data)

    # Download Report
    report = df.describe(include="all").to_csv()

    st.download_button(
        label="📥 Download Analysis Report",
        data=report,
        file_name="analysis_report.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to begin analysis.")