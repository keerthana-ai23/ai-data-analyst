import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("🤖 AI Data Analyst")

st.write(
    "Upload any CSV file and automatically perform EDA."
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Duplicate Records")
    st.write(df.duplicated().sum())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    st.subheader("AI Insights")

    missing = df.isnull().sum().sum()

    st.write(
        f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    st.write(
        f"Total missing values: {missing}"
    )

    duplicates = df.duplicated().sum()

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

    report = df.describe().to_csv()

    st.download_button(
        label="Download Analysis Report",
        data=report,
        file_name="analysis_report.csv",
        mime="text/csv"
    )
