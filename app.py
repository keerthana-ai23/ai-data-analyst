import streamlit as st
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("🤖 AI Data Analyst Assistant")

st.write(
    "Upload any CSV file and perform Exploratory Data Analysis."
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # DATA PREVIEW
    # -----------------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # SHAPE
    # -----------------------------
    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # -----------------------------
    # MISSING VALUES
    # -----------------------------
    st.subheader("Missing Values")
    missing_df = pd.DataFrame(
        df.isnull().sum(),
        columns=["Missing Count"]
    )
    st.dataframe(missing_df)

    # -----------------------------
    # DUPLICATES
    # -----------------------------
    duplicates = df.duplicated().sum()

    st.subheader("Duplicate Records")
    st.write(duplicates)

    # -----------------------------
    # DATA TYPES
    # -----------------------------
    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    # -----------------------------
    # SUMMARY STATS
    # -----------------------------
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numeric_columns) > 0:

        st.subheader("Visualization")

        selected_column = st.selectbox(
            "Choose Numeric Column",
            numeric_columns
        )

        st.bar_chart(
            df[selected_column]
            .value_counts()
            .head(20)
        )

    # -----------------------------
    # AI INSIGHTS
    # -----------------------------
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

    # -----------------------------
    # DATA CLEANING SUGGESTIONS
    # -----------------------------
    st.subheader("🧹 Data Cleaning Suggestions")

    if missing > 0:
        st.warning(
            "Consider filling or removing missing values."
        )
    else:
        st.success(
            "No missing values detected."
        )

    if duplicates > 0:
        st.warning(
            f"{duplicates} duplicate rows detected. Consider removing duplicates."
        )
    else:
        st.success(
            "No duplicate rows detected."
        )

    # -----------------------------
    # SMART DATASET Q&A
    # -----------------------------
    st.subheader("💬 Ask AI About Your Dataset")

    user_question = st.text_input(
        "Ask any question about your dataset"
    )

    if user_question:

        q = user_question.lower()

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
                f"Total missing values: {missing}"
            )

        elif "duplicate" in q:
            st.success(
                f"Duplicate rows: {duplicates}"
            )

        elif "datatype" in q or "data type" in q:
            st.dataframe(df.dtypes.astype(str))

        elif "summary" in q or "statistics" in q:
            st.dataframe(df.describe())

        elif "clean" in q:
            if missing == 0:
                st.success(
                    "Dataset appears clean with no missing values."
                )
            else:
                st.warning(
                    "Dataset requires cleaning due to missing values."
                )

        elif "tell" in q or "explain" in q:

            st.success(
                f"""
                This dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

                There are {missing} missing values and {duplicates} duplicate rows.

                The dataset can be used for exploratory data analysis,
                trend analysis, machine learning, and business insights.
                """
            )

        else:

            st.info(
                f"""
                I can answer questions about:

                • Rows
                • Columns
                • Missing values
                • Duplicate records
                • Summary statistics
                • Data types
                • Cleaning suggestions
                """
            )

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
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