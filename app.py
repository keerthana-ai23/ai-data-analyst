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

        st.bar_chart(
            df[selected_col].value_counts().head(20)
        )

    # AI Assistant
    st.subheader("🤖 Ask AI About Your Dataset")

    question = st.text_input(
        "Ask a question about the uploaded dataset"
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
                f"The dataset contains {total_missing} missing values."
            )

        elif "duplicate" in q:
            st.success(
                f"The dataset contains {duplicates} duplicate rows."
            )

        elif "datatype" in q or "data type" in q:
            st.dataframe(df.dtypes.astype(str))

        elif "average" in q or "mean" in q:

            numeric_df = df.select_dtypes(include="number")

            if len(numeric_df.columns) > 0:
                st.dataframe(
                    numeric_df.mean().reset_index().rename(
                        columns={
                            "index": "Column",
                            0: "Mean"
                        }
                    )
                )

        elif "summary" in q:
            st.dataframe(df.describe())

        elif "clean" in q:
            st.write(
                "Recommended cleaning steps:"
            )

            if total_missing > 0:
                st.write(
                    f"• Handle {total_missing} missing values."
                )

            if duplicates > 0:
                st.write(
                    f"• Remove {duplicates} duplicate rows."
                )

            if total_missing == 0 and duplicates == 0:
                st.write(
                    "• Dataset appears clean."
                )

        else:
            st.info(
                "I can currently answer questions about rows, columns, missing values, duplicates, averages, summary statistics, data types, and cleaning suggestions."
            )

    # Download Report
    report = df.describe(include="all").to_csv()

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