import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    # =============================
# CORRELATION ANALYSIS
# =============================

st.subheader("📊 Correlation Analysis")

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

if len(numeric_df.columns) > 1:

    corr_matrix = numeric_df.corr()

    st.dataframe(corr_matrix)

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
# =============================
# HEALTHCARE AI INSIGHTS
# =============================

st.subheader("🩺 Healthcare AI Insights")

missing = df.isnull().sum().sum()

st.write(
    f"Total Records: {df.shape[0]}"
)

st.write(
    f"Features: {df.shape[1]}"
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

if len(numeric_df.columns) > 1:

    corr_matrix = numeric_df.corr()

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

    strongest_value = strongest_corr.iloc[0]

    strongest_pair = strongest_corr.index[0]

    st.info(
        f"""
        Strongest relationship detected:

        {strongest_pair[0]}
        ↔
        {strongest_pair[1]}

        Correlation:
        {strongest_value:.2f}
        """
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
# =============================
# DATASET ASSISTANT
# =============================

st.subheader(
    "💬 Ask About Dataset"
)

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

        st.dataframe(
            corr_matrix
        )

    elif (
        "explain" in q
        or "dataset" in q
    ):

        st.success(
            f"""
            This healthcare dataset contains
            {df.shape[0]} records and
            {df.shape[1]} features.

            Missing values:
            {missing}

            Duplicate records:
            {duplicates}

            The dataset can be used for:

            • Patient risk analysis

            • Disease prediction

            • Healthcare trend analysis

            • Machine learning

            • Clinical decision support
            """
        )

    elif (
        "risk" in q
        or "high risk" in q
    ):

        st.info(
            """
            High-risk patients can be identified
            using highly correlated variables
            and predictive modeling.
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

            • What insights do you find?
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