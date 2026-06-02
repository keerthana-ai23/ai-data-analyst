import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("🤖 AI Data Analyst")
st.write("Upload any CSV file and automatically perform EDA.")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Choose a column for visualization",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_col,
            title=f"Distribution of {selected_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a CSV file to begin analysis.")