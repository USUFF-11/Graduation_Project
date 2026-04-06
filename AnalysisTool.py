import streamlit as st
import pandas as pd
from openai import OpenAI

# =========================
# 🔐 OPENROUTER CONFIG
# =========================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="Api_Key_Here"
)

st.set_page_config(page_title="AI Data Tool", layout="wide")

st.markdown("""
<style>
    /* Background */
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    /* Title */
    h1 {
        color: #7c83fd;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    /* Subheaders */
    h2, h3 {
        color: #a0a8ff;
        border-bottom: 1px solid #2a2d3e;
        padding-bottom: 6px;
        margin-top: 1.5rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1e2130;
        color: #c9d1ff;
        border: 1px solid #3a3f5c;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 0.85rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #7c83fd;
        color: white;
        border-color: #7c83fd;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: #1e2130;
        color: #e0e0e0;
        border: 1px solid #3a3f5c;
        border-radius: 10px;
        padding: 10px 14px;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #1e2130;
        border: 1px dashed #3a3f5c;
        border-radius: 12px;
        padding: 10px;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Success box */
    .stSuccess {
        background-color: #1a2e1a;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #7c83fd !important;
    }

    /* Divider */
    hr {
        border-color: #2a2d3e;
    }
</style>
""", unsafe_allow_html=True)

st.title("✦ AI Data Analysis Tool")

uploaded_file = st.file_uploader("📂 Upload your data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file,index_col = 0)

    # =========================
    # 🧹 CLEANING
    # =========================

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip().str.capitalize().str.replace(" ", "_")

    cleaning_report = []

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates()
    if duplicates_count > 0:
        cleaning_report.append(f"🗑️ Removed **{duplicates_count}** duplicate rows.")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col].fillna(df[col].median(), inplace=True)
            cleaning_report.append(f"🔢 `{col}`: filled **{missing}** missing values with median.")

    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col].fillna("Unknown", inplace=True)
            cleaning_report.append(f"🔤 `{col}`: filled **{missing}** missing values with 'Unknown'.")

    # =========================
    # 📊 SHOW DATA
    # =========================

    st.subheader("📊 Cleaned Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧹 Cleaning Report")
    if cleaning_report:
        for item in cleaning_report:
            st.markdown(item)
    else:
        st.success("✅ Data is clean! No issues found.")


    # =========================
    # 🧠 CONTEXT
    # =========================

    summary = df.describe().to_string()
    columns = ", ".join(df.columns)
    sample_data = df.head(20).to_string()

    context = f"""
    Dataset Columns:
    {columns}

    Summary:
    {summary}

    Sample:
    {sample_data}
    """

    # =========================
    # 🤖 GENERATE QUESTIONS
    # =========================

    if "questions_list" not in st.session_state:
        with st.spinner("Generating smart questions..."):
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 5 short business questions based on the dataset. One question per line."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            )
        generated_questions = response.choices[0].message.content
        import re
        st.session_state.questions_list = [re.sub(r'^\d+[\.\)]\s*', '', q.strip("- ").strip()) for q in generated_questions.split("\n") if q.strip()]

    questions_list = st.session_state.questions_list

    # =========================
    # 💡 BUTTONS
    # =========================

    st.subheader("💡 AI Suggested Questions")

    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

    cols = st.columns(2)

    for i, q in enumerate(questions_list):
        if cols[i % 2].button(q):
            st.session_state.selected_question = q

    # =========================
    # 💬 INPUT
    # =========================

    st.subheader("💬 Ask about your data")

    typed_question = st.text_input("Type your question or pick one above", value=st.session_state.selected_question or "")

    if typed_question != (st.session_state.selected_question or ""):
        st.session_state.selected_question = None

    question = typed_question

    # =========================
    # 🤖 AI ANSWER
    # =========================

    if question:
        with st.spinner("Analyzing..."):

            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. The user gave you a dataset. Answer the question directly using the actual data provided. Give a specific number or fact as the answer first, then a brief explanation if needed. Do NOT say you don't have access to the full dataset. Do NOT suggest how to find the answer. Just answer it."
                    },
                    {
                        "role": "user",
                        "content": context + "\n\nQuestion: " + question
                    }
                ]
            )

            answer = response.choices[0].message.content

        st.subheader("🤖 Answer")
        st.write(answer)
