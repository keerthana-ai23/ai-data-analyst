import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Healthcare AI Data Analyst",
    layout="wide"
)

st.title("🩺 Healthcare AI Data Analyst Assistant")

st.write(
    "Upload any healthcare CSV dataset and perform Exploratory Data Analysis."
)

# ==================================
# FILE UPLOAD
# ==================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# ==================================
# MAIN APP
# ==================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ==================================
    # DATA PREVIEW
    # ==================================

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ==================================
    # DATASET SHAPE
    # ==================================

    st.subheader("📏 Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # ==================================
    # DATA TYPES
    # ==================================

    st.subheader("🔠 Data Types")
    st.dataframe(df.dtypes.astype(str))

    # ==================================
    # MISSING VALUES
    # ==================================

    st.subheader("❌ Missing Values")

    missing_df = pd.DataFrame(
        df.isnull().sum(),
        columns=["Missing Count"]
    )

    st.dataframe(missing_df)

    missing = df.isnull().sum().sum()

    # ==================================
    # DUPLICATES
    # ==================================

    duplicates = df.duplicated().sum()

    st.subheader("📑 Duplicate Records")
    st.write(duplicates)

    # ==================================
    # SUMMARY STATISTICS
    # ==================================

    st.subheader("📊 Summary Statistics")

    st.dataframe(
        df.describe(include="all")
    )

    # ==================================
    # CORRELATION ANALYSIS
    # ==================================

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    if len(numeric_df.columns) > 1:

        st.subheader("📈 Correlation Matrix")

        corr_matrix = numeric_df.corr()

        st.dataframe(corr_matrix)

        # ==================================
        # HEATMAP
        # ==================================

        st.subheader("🔥 Correlation Heatmap")

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        heatmap = ax.imshow(
            corr_matrix,
            cmap="coolwarm",
            aspect="auto"
        )

        ax.set_xticks(
            range(len(corr_matrix.columns))
        )

        ax.set_xticklabels(
            corr_matrix.columns,
            rotation=90
        )

        ax.set_yticks(
            range(len(corr_matrix.columns))
        )

        ax.set_yticklabels(
            corr_matrix.columns
        )

        plt.colorbar(heatmap)

        st.pyplot(fig)

        # ==================================
        # STRONGEST CORRELATION
        # ==================================

        strongest_corr = (
            corr_matrix.abs()
            .unstack()
            .sort_values(
                ascending=False
            )
        )

        strongest_corr = strongest_corr[
            strongest_corr < 1
        ]

        if len(strongest_corr) > 0:

            strongest_pair = strongest_corr.index[0]
            strongest_value = strongest_corr.iloc[0]

            st.info(
                f"""
                Strongest relationship found:

                {strongest_pair[0]} ↔ {strongest_pair[1]}

                Correlation Score:
                {strongest_value:.2f}
                """
            )

    # ==================================
    # VISUALIZATION
    # ==================================

    numeric_columns = numeric_df.columns

    if len(numeric_columns) > 0:

        st.subheader("📉 Data Visualization")

        selected_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        st.bar_chart(
            df[selected_column]
            .value_counts()
            .head(20)
        )

    # ==================================
    # HEALTHCARE AI INSIGHTS
    # ==================================

    st.subheader("🩺 Healthcare AI Insights")

    st.write(
        f"Total Records: {df.shape[0]}"
    )

    st.write(
        f"Total Features: {df.shape[1]}"
    )

    st.write(
        f"Missing Values: {missing}"
    )

    st.write(
        f"Duplicate Records: {duplicates}"
    )

    if missing == 0:
        st.success(
            "Dataset quality is good."
        )
    else:
        st.warning(
            "Dataset contains missing values."
        )

    if duplicates > 0:
        st.warning(
            f"{duplicates} duplicate records detected."
        )

    # ==================================
    # CLEANING SUGGESTIONS
    # ==================================

    st.subheader("🧹 Data Cleaning Suggestions")

    if missing > 0:
        st.warning(
            "Consider handling missing values."
        )
    else:
        st.success(
            "No missing values detected."
        )

    if duplicates > 0:
        st.warning(
            "Consider removing duplicate rows."
        )
    else:
        st.success(
            "No duplicate rows detected."
        )

    # ==================================
    # DATASET ASSISTANT
    # ==================================

    st.subheader("💬 Ask About Your Dataset")

    question = st.text_input(
        "Ask a question"
    )

    if question:

        q = question.lower()

        if "row" in q:

            st.success(
                f"Dataset contains {df.shape[0]} rows."
            )

        elif "column" in q:

            st.success(
                f"Dataset contains {df.shape[1]} columns."
            )

        elif "missing" in q:

            st.success(
                f"Missing values: {missing}"
            )

        elif "duplicate" in q:

            st.success(
                f"Duplicate records: {duplicates}"
            )

        elif "correlation" in q:

            if len(numeric_df.columns) > 1:
                st.dataframe(corr_matrix)
            else:
                st.warning(
                    "Not enough numeric columns."
                )

        elif "explain" in q or "dataset" in q:

            st.success(
                f"""
                This dataset contains
                {df.shape[0]} records and
                {df.shape[1]} features.

                Missing values: {missing}

                Duplicate records: {duplicates}

                Potential healthcare uses:

                • Disease prediction

                • Patient risk analysis

                • Healthcare trend analysis

                • Clinical decision support

                • Predictive analytics
                """
            )

        else:

            st.info(
                """
                Try asking:

                • Explain dataset

                • How many rows?

                • How many columns?

                • Missing values

                • Duplicate records

                • Show correlations
                """
            )

    # ==================================
    # DOWNLOAD REPORT
    # ==================================

    report = df.describe(
        include="all"
    ).to_csv()

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