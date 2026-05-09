import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import re
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# =========================
# 🔐 OPENROUTER CONFIG
# =========================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="API KEY HERE"
)

def ask_ai(system_prompt, user_prompt, stream=False):
    if stream:
        # Streaming mode — yields text chunks and prints reasoning tokens at the end
        full_response = ""
        reasoning_tokens = None
        with client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            stream_options={"include_usage": True},
        ) as stream_resp:
            for chunk in stream_resp:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_response += delta.content
                # Capture usage/reasoning tokens from the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
                    if hasattr(usage, "completion_tokens_details"):
                        details = usage.completion_tokens_details
                        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        return full_response, reasoning_tokens
    else:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

st.set_page_config(page_title="AI Data Tool", layout="wide", page_icon="ChatGPT Image May 9, 2026, 11_21_34 PM.png")

# Hide Streamlit default UI elements
st.markdown("""
    <style>
        header[data-testid="stHeader"] { display: none !important; }
        #MainMenu { display: none !important; }
        footer { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="manage-app-button"] { display: none !important; }
        .stDeployButton { display: none !important; }
    </style>
""", unsafe_allow_html=True)

def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("Style.css")
# =========================
# ⚡ CACHED FUNCTIONS
# =========================

@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    import io
    file_obj = io.BytesIO(file_bytes)
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > 50:
        chunks = []
        for chunk in pd.read_csv(file_obj, chunksize=50000, index_col=0):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.read_csv(file_obj, index_col=0)

@st.cache_data(show_spinner=False)
def get_meaningful_missing_cols(date_cols, columns_context):
    """Ask AI which date columns should keep their missing values as meaningful."""
    if not date_cols:
        return []
    system = """You are a data analyst. Given a list of date column names, decide which ones have MEANINGFUL missing values 
(i.e. missing = event hasn't happened yet, like delivery not done, shipment not sent, etc.)
vs columns where missing = data error (like birth date, order date, created date).

Return ONLY a comma-separated list of column names that should keep their missing values. No explanation."""
    user = f"All columns in dataset: {columns_context}\nDate columns to analyze: {', '.join(date_cols)}"
    try:
        result = ask_ai(system, user)
        return [c.strip() for c in result.split(",") if c.strip() in date_cols]
    except:
        return []

def clean_dataframe(df):
    cleaning_report = []

    # ── 1. Remove unnamed columns ──────────────────────────────
    unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # ── 2. Standardize column names ────────────────────────────
    df.columns = df.columns.str.strip().str.capitalize().str.replace(r'\s+', '_', regex=True)

    # ── 3. Remove constant columns ─────────────────────────────
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        df.drop(columns=const_cols, inplace=True)
        cleaning_report.append(f"🗑️ Removed **{len(const_cols)}** constant column(s): {', '.join(f'`{c}`' for c in const_cols)}")

    # ── 4. Smart type detection by column name ─────────────────
    date_kw    = ["date", "time", "birth", "created", "updated", "timestamp", "dt"]
    numeric_kw = ["price", "cost", "salary", "amount", "revenue", "profit", "fee", "rate", "score",
                  "qty", "quantity", "total", "weight", "height", "age", "distance", "duration", "units",
                  "year", "month", "day"]
    bool_kw    = ["is_", "has_", "flag", "active", "enabled", "returned", "damaged"]

    sample = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

    # Pre-detect date columns
    potential_date_cols = [c for c in df.columns if df[c].dtype == object and any(k in c.lower() for k in date_kw)]

    for col in list(df.columns):
        if df[col].dtype != object:
            continue
        col_lower = col.lower()

        # Date
        if any(k in col_lower for k in date_kw):
            try:
                converted = pd.to_datetime(sample[col], infer_datetime_format=True, errors="coerce")
                if converted.notna().mean() > 0.7:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce").dt.date
                    missing_dates = df[col].isnull().sum()
                    cleaning_report.append(f"📅 `{col}`: converted to **date**." + (f" (**{missing_dates}** missing kept.)" if missing_dates > 0 else ""))
                    continue
            except: pass

        # Numeric
        if any(k in col_lower for k in numeric_kw):
            try:
                cleaned = sample[col].astype(str).str.replace(r'[$,€£\s]', '', regex=True)
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().mean() > 0.7:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[$,€£\s]', '', regex=True), errors="coerce")
                    cleaning_report.append(f"🔢 `{col}`: converted to **numeric**.")
                    continue
            except: pass

        # Boolean
        if any(k in col_lower for k in bool_kw):
            try:
                bool_map = {"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False}
                if sample[col].astype(str).str.lower().isin(bool_map).mean() > 0.7:
                    df[col] = df[col].astype(str).str.lower().map(bool_map)
                    cleaning_report.append(f"✅ `{col}`: converted to **boolean**.")
                    continue
            except: pass

        # Default numeric attempt
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except: pass

    # ── 5. Duplicates ──────────────────────────────────────────
    dup_count = df.duplicated().sum()
    id_kw = ["id", "code", "key", "uid", "ref", "sku"]
    id_candidates = [c for c in df.columns if any(k == c.lower() or c.lower().endswith(k) or c.lower().startswith(k) for k in id_kw)]
    id_col = max(id_candidates, key=lambda c: df[c].nunique()) if id_candidates else None
    df = df.drop_duplicates()
    if dup_count > 0:
        base = f"🗑️ Removed **{dup_count}** duplicate rows"
        cleaning_report.append(f"{base} based on `{id_col}`." if id_col else f"{base}.")

    total_rows = len(df)

    # ── 6. Numeric: missing values + outlier capping ───────────
    for col in list(df.select_dtypes(include=['int64', 'float64']).columns):
        missing = df[col].isnull().sum()
        if missing > 0:
            pct = missing / total_rows
            if pct > 0.6:
                df.drop(columns=[col], inplace=True)
                cleaning_report.append(f"🗑️ `{col}`: dropped — **{pct*100:.0f}%** missing.")
                continue
            fill_val = df[col].mean() if pct > 0.05 else df[col].median()
            method   = "mean" if pct > 0.05 else "median"
            df[col].fillna(fill_val, inplace=True)
            cleaning_report.append(f"🔢 `{col}`: filled **{missing}** missing with {method}.")

        # IQR outlier capping
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                cleaning_report.append(f"✂️ `{col}`: capped **{outliers}** outliers using IQR.")

    # ── 7. Categorical: text consistency + missing ─────────────
    for col in list(df.select_dtypes(include=['object']).columns):
        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

        # Normalize common variants
        norm_map = {
            r'^u\.?s\.?a?\.?$': 'USA', r'^united states.*': 'USA',
            r'^u\.?k\.?$': 'UK',       r'^united kingdom.*': 'UK',
            r'^nan$': None, r'^none$': None, r'^n/a$': None,
            r'^-$': None,  r'^$': None
        }
        for pattern, val in norm_map.items():
            mask = df[col].str.lower().str.fullmatch(pattern.strip('^$'), na=False) if '.*' not in pattern \
                   else df[col].str.lower().str.match(pattern, na=False)
            if mask.any():
                df.loc[mask, col] = val

        missing = df[col].isnull().sum()
        if missing > 0:
            pct = missing / total_rows
            if pct > 0.6:
                df.drop(columns=[col], inplace=True)
                cleaning_report.append(f"🗑️ `{col}`: dropped — **{pct*100:.0f}%** missing.")
            else:
                cleaning_report.append(f"⚠️ `{col}`: **{missing}** missing values kept as-is.")

    return df, cleaning_report

@st.cache_data(show_spinner=False)
def get_context(df):
    summary = df.describe().round(2).to_string()
    columns = ", ".join(df.columns)
    sample_data = df.sample(min(5, len(df))).to_string()
    return summary, columns, sample_data

# =========================
# 📄 PAGE ROUTING
# =========================
if "page" not in st.session_state:
    st.session_state.page = "main"

# Splash screen on first load

if "app_loaded" not in st.session_state:
    st.markdown("""
    <div id="splash">
      <div class="splash-title">✦ AI Data Analysis Tool</div>
      <div class="splash-sub">Your AI-powered Data Analyst</div>
      <div class="splash-bar"><div class="splash-fill"></div></div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.app_loaded = True

import os as _os
_LOGO_PATH = next((_f for _f in _os.listdir(".") if _f.lower().endswith(".png") and _f != "logo.png"), None)
if _LOGO_PATH and _os.path.exists(_LOGO_PATH):
    import base64 as _b64
    with open(_LOGO_PATH, "rb") as _f:
        _logo_b64 = _b64.b64encode(_f.read()).decode()
    st.markdown(f"""
        <!-- Watermark in center -->
        <div style="position:fixed; top:50%; left:50%; transform:translate(-50%, -50%); z-index:0; pointer-events:none;">
            <img src="data:image/png;base64,{_logo_b64}" style="width:1200px; height:1200px; object-fit:contain; opacity:0.08; filter: drop-shadow(0px 0px 30px rgba(31,170,138,0.9)) drop-shadow(0px 0px 60px rgba(31,170,138,0.6)) drop-shadow(0px 0px 100px rgba(31,170,138,0.4));">
        </div>
    """, unsafe_allow_html=True)

st.title("✦ AI Data Analysis Tool")

uploaded_file = st.file_uploader("📂 Upload your data", type=["csv"])

# If file is removed, clear everything
if uploaded_file is None and "uploaded_file" in st.session_state:
    keys_to_clear = [
        "uploaded_file", "last_file_name", "last_file_size",
        "questions_list", "dashboard_kpis", "dashboard_charts", "build_dashboard",
        "user_path", "df_for_dashboard", "selected_question",
        "df_for_ml", "ml_results", "ml_target_col", "ml_chat", "ml_trained",
        "ml_leaderboard", "last_answer", "last_question", "last_reasoning_tokens",
        "app_loaded"
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    st.session_state.page = "main"
    st.rerun()

if uploaded_file:
    # مسح الـ cache لو الملف اتغير
    prev_name = st.session_state.get("last_file_name", None)
    prev_size = st.session_state.get("last_file_size", None)
    if prev_name != uploaded_file.name or prev_size != uploaded_file.size:
        keys_to_clear = ["questions_list",
                         "dashboard_kpis", "dashboard_charts", "build_dashboard",
                         "user_path", "df_for_dashboard", "selected_question",
                         "df_for_ml", "ml_results", "ml_target_col"]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.session_state.last_file_name = uploaded_file.name
        st.session_state.last_file_size = uploaded_file.size
    st.session_state.uploaded_file = uploaded_file

if st.session_state.page == "main" and "uploaded_file" in st.session_state and st.session_state.uploaded_file:
    uploaded_file = st.session_state.uploaded_file

    file_size_mb = uploaded_file.size / (1024 * 1024)

    # Load & Clean
    file_bytes = uploaded_file.getvalue()
    with st.spinner("Processing your data..."):
        raw_df = load_file(file_bytes, uploaded_file.name)
        df, cleaning_report = clean_dataframe(raw_df)

    st.markdown(f"""
        <div style="
            display: flex; align-items: center; gap: 16px;
            background: linear-gradient(135deg, rgba(31,170,138,0.15), rgba(11,58,47,0.3));
            border: 1px solid rgba(31,170,138,0.4);
            border-radius: 14px; padding: 16px 24px; margin: 12px 0;
        ">
            <div>
                <div style="color: #7DE6B0; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;">Data Loaded Successfully</div>
                <div style="color: #E6F1EC; font-size: 1.4rem; font-weight: 700; margin-top: 2px;">
                    {len(df):,} rows &nbsp;×&nbsp; {len(df.columns)} columns
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


    # =========================
    # 📊 SHOW DATA
    # =========================

    st.markdown('<div class="section-title">📊 Cleaned Data</div>', unsafe_allow_html=True)
    st.caption(f"{len(df)} rows × {len(df.columns)} columns — showing first 500 rows")
    # Display with 1 decimal for float columns, without changing actual data
    display_df = df.head(500).copy()
    for col in display_df.select_dtypes(include='float').columns:
        display_df[col] = display_df[col].round(1)
    st.dataframe(display_df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Cleaned Data", data=csv, file_name="cleaned_data.csv", mime="text/csv")

    st.subheader("🧹 Cleaning Report")
    if cleaning_report:
        import re

        def clean_text(text):
            text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]+', '', text)
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = text.replace('`', '')
            return text.strip()

        def extract_col(text):
            # Extract column name (first word-like token before colon)
            m = re.search(r'([A-Za-z_][A-Za-z0-9_]*)\s*:', clean_text(text))
            return m.group(1) if m else clean_text(text)

        # ── Categorize ──────────────────────────────────────────
        converted_date    = [i for i in cleaning_report if "converted to" in i.lower() and "date" in i.lower()]
        converted_numeric = [i for i in cleaning_report if "converted to" in i.lower() and "numeric" in i.lower()]
        converted_bool    = [i for i in cleaning_report if "converted to" in i.lower() and "boolean" in i.lower()]

        removed_dup  = [i for i in cleaning_report if "duplicate" in i.lower()]
        removed_col  = [i for i in cleaning_report if ("dropped" in i.lower() or ("removed" in i.lower() and "duplicate" not in i.lower()))]

        filled_median  = [i for i in cleaning_report if "median" in i.lower()]
        filled_mean    = [i for i in cleaning_report if "mean" in i.lower()]
        filled_mode    = [i for i in cleaning_report if "most common" in i.lower()]
        filled_outlier = [i for i in cleaning_report if "capped" in i.lower() or "outlier" in i.lower()]

        def render_subgroup(label, items):
            if not items:
                return ""
            cols_html = "".join(
                f'<span style="display:inline-block; background:rgba(31,170,138,0.12); '
                f'border:1px solid rgba(31,170,138,0.25); border-radius:6px; '
                f'padding:2px 10px; margin:3px 3px 3px 0; color:#c8e6d4; font-size:0.78rem;">'
                f'{extract_col(i)}</span>'
                for i in items
            )
            return f"""
                <div style="margin-bottom:10px;">
                    <div style="color:#7DE6B0; font-size:0.72rem; letter-spacing:1px;
                                text-transform:uppercase; margin-bottom:5px;">{label}</div>
                    <div>{cols_html}</div>
                </div>
            """

        def render_card(title, subgroups):
            body = "".join(render_subgroup(lbl, items) for lbl, items in subgroups)
            if not body.strip().replace('<div style="margin-bottom:10px;"></div>', ''):
                body = '<div style="color:#4a7a62; font-size:0.84rem;">No changes</div>'
            return f"""
                <div style="background:rgba(15,35,28,0.6); border:1px solid rgba(31,170,138,0.25);
                            border-radius:12px; padding:18px 20px; min-height:120px;">
                    <div style="color:#E6F1EC; font-size:0.75rem; font-weight:600;
                                letter-spacing:1.5px; text-transform:uppercase; margin-bottom:14px;">
                        {title}
                    </div>
                    {body}
                </div>
            """

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(render_card("Data Type Converted", [
                ("Date",    converted_date),
                ("Numeric", converted_numeric),
                ("Boolean", converted_bool),
            ]), unsafe_allow_html=True)

        with col2:
            st.markdown(render_card("Removed", [
                ("Duplicate Rows", removed_dup),
                ("Columns",        removed_col),
            ]), unsafe_allow_html=True)

        with col3:
            st.markdown(render_card("Filled Missing Values", [
                ("Median",         filled_median),
                ("Mean",           filled_mean),
                ("Most Common",    filled_mode),
                ("Capped Outliers",filled_outlier),
            ]), unsafe_allow_html=True)

    else:
        st.success("Data is clean! No issues found.")
    # =========================
    # 🛤️ PATH SELECTION
    # =========================

    # =========================
    # 🧠 CONTEXT (cached)
    # =========================

    summary, columns, sample_data = get_context(df)

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
            generated_questions = ask_ai(
                system_prompt="Generate 5 short business questions based on the dataset. One question per line.",
                user_prompt=context
            )
        st.session_state.questions_list = [re.sub(r'^\d+[\.\)]\s*', '', q.strip("- ").strip()) for q in generated_questions.split("\n") if q.strip()]

    questions_list = st.session_state.questions_list

    # =========================
    # 💡 BUTTONS
    # =========================

    st.subheader("💡 Suggested Questions")

    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

    cols = st.columns(2)

    for i, q in enumerate(questions_list):
        # Clean backticks and markdown from question display
        clean_q = re.sub(r'`([^`]+)`', r'\1', q)
        if cols[i % 2].button(clean_q, key=f"q_btn_{i}"):
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

    ask_clicked = st.button("Ask", key="ask_btn")

    if ask_clicked and question:
        with st.spinner("Analyzing..."):
            q_lower = question.lower()
            pre_computed = ""

            for col in df.columns:
                if col.lower() in q_lower:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        pre_computed += (
                            f"{col} — sum: {df[col].sum():,.2f}, "
                            f"mean: {df[col].mean():,.2f}, "
                            f"min: {df[col].min():,.2f}, "
                            f"max: {df[col].max():,.2f}, "
                            f"count: {df[col].count():,}\n"
                        )
                    else:
                        vc = df[col].value_counts().head(15).to_string()
                        pre_computed += f"{col} value counts:\n{vc}\n"

            # Group-by analysis: if question mentions two columns, compute groupby
            mentioned_cols = [c for c in df.columns if c.lower() in q_lower]
            if len(mentioned_cols) >= 2:
                cat_col = next((c for c in mentioned_cols if df[c].dtype == object), None)
                num_col = next((c for c in mentioned_cols if pd.api.types.is_numeric_dtype(df[c])), None)
                if cat_col and num_col:
                    grp = df.groupby(cat_col)[num_col].agg(['sum','mean','count']).round(2).to_string()
                    pre_computed += f"\nGrouped {num_col} by {cat_col}:\n{grp}\n"

            # Date range filtering
            import re as _re
            date_pattern = _re.findall(r'\d{1,2}/\d{1,2}/\d{4}', question)
            if len(date_pattern) >= 2:
                try:
                    d1 = pd.to_datetime(date_pattern[0])
                    d2 = pd.to_datetime(date_pattern[1])
                    for col in df.columns:
                        try:
                            parsed = pd.to_datetime(df[col], errors='coerce')
                            if parsed.notna().mean() > 0.7:
                                mask = (parsed >= d1) & (parsed <= d2)
                                filtered = df[mask]
                                pre_computed += f"\nRows where {col} between {date_pattern[0]} and {date_pattern[1]}: {len(filtered):,} rows\n"
                                pre_computed += filtered.head(20).to_string() + "\n"
                                break
                        except: pass
                except: pass

            targeted_context = f"Dataset: {len(df):,} rows × {len(df.columns)} columns\nColumns: {', '.join(df.columns)}\n\n"
            targeted_context += f"Numeric summary:\n{df.describe().round(2).to_string()}\n\n"
            if pre_computed:
                targeted_context += f"Computed data for this question:\n{pre_computed}\n"

            answer, reasoning_tokens = ask_ai(
                system_prompt="""You are a data analyst. Answer in maximum 3 sentences. Be direct — start with the actual number or fact from the computed data provided. Do NOT say you need more data. No intros, no bullet points.""",
                user_prompt=targeted_context + "\n\nQuestion: " + question,
                stream=True
            )
            st.session_state.last_answer = answer
            st.session_state.last_question = question
            st.session_state.last_reasoning_tokens = reasoning_tokens
    elif not question:
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_question", None)

    if "last_answer" in st.session_state and st.session_state.get("last_question") == question and question:
        clean_answer = st.session_state.last_answer.replace("\\$", "$").replace("\\_", "_").replace("\\*", "*")
        reasoning_tokens = st.session_state.get("last_reasoning_tokens")
        st.markdown(f"""
            <div style="
                background: rgba(15,35,28,0.6);
                border: 1px solid rgba(31,170,138,0.25);
                border-radius: 12px;
                padding: 18px 22px;
                margin-top: 12px;
                color: #c8e6d4;
                font-size: 0.95rem;
                line-height: 1.7;
            ">
                {clean_answer.replace(chr(10), '<br>')}
            </div>
        """, unsafe_allow_html=True)

    # =========================
    # 📊 GENERATE DASHBOARD BTN
    # =========================
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("📊 Generate a Dashboard", use_container_width=True, key="go_dashboard"):
             st.session_state.df_for_dashboard = df
             st.session_state.page = "dashboard"
             st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 ML Prediction Studio", use_container_width=True, key="go_ml"):
            st.session_state.df_for_ml = df
            st.session_state.page = "ml"
            st.rerun()

# =========================
# 📊 DASHBOARD PAGE
# =========================
elif st.session_state.page == "dashboard":

    if st.button("← Back to Analysis"):
        st.session_state.page = "main"
        st.session_state.user_path = None
        st.session_state.build_dashboard = False
        st.session_state.dashboard_kpis = None
        st.session_state.dashboard_charts = None
        st.session_state.kpi_count = 2
        st.rerun()

    st.title("📊 AI Dashboard Builder")

    df = st.session_state.get("df_for_dashboard", None)
    if df is None:
        st.warning("No data found. Please go back and upload a file.")
        st.stop()

    numeric_cols_db = list(df.select_dtypes(include=["int64", "float64"]).columns)
    all_cols_db = list(df.columns)
    cat_cols_db = list(df.select_dtypes(include=["object"]).columns)
    summary, columns, sample_data = get_context(df)

    # Step 1: Ask AI for KPI suggestions
    if "dashboard_kpis" not in st.session_state or st.session_state.dashboard_kpis is None:
        with st.spinner("AI is suggesting KPIs for your data..."):
            st.session_state.dashboard_kpis = ask_ai(
                system_prompt="""You are a BI analyst. Suggest exactly 5 KPIs for this dataset.
Return in this strict format:
KPI: <kpi name>
METRIC: <sum|mean|count|max|min>
COLUMN: <exact column name>
LABEL: <short display label>
---
Repeat 5 times. No extra text.""",
                user_prompt=f"Columns: {columns}\nSummary:\n{summary}"
            )
    # Parse KPIs
    kpi_blocks = [b.strip() for b in st.session_state.dashboard_kpis.split("---") if b.strip()]
    kpis = []
    for block in kpi_blocks:
        kpi_lines = {l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip() for l in block.split("\n") if ":" in l}
        kpis.append(kpi_lines)

    # Step 2: Let user confirm/edit KPIs
    st.subheader("✅ Confirm your KPIs")
    st.caption("AI suggested 2 KPIs — add more if you need.")

    if "kpi_count" not in st.session_state:
        st.session_state.kpi_count = 2

    confirmed_kpis = []
    kpi_cols = st.columns(st.session_state.kpi_count)
    for i in range(st.session_state.kpi_count):
        with kpi_cols[i]:
            kpi = kpis[i] if i < len(kpis) else {}
            default_col = kpi.get("COLUMN", numeric_cols_db[0] if numeric_cols_db else "")
            default_metric = kpi.get("METRIC", "sum")
            default_label = kpi.get("LABEL", f"KPI {i+1}")
            label_inp = st.text_input("Label", value=default_label, key=f"kpi_label_{i}")
            col_sel = st.selectbox("Column",
                                   options=numeric_cols_db if numeric_cols_db else [""],
                                   index=numeric_cols_db.index(default_col) if default_col in numeric_cols_db else 0,
                                   key=f"kpi_col_{i}")
            metric_sel = st.selectbox("Metric", ["sum", "mean", "count", "max", "min"],
                                      index=["sum", "mean", "count", "max", "min"].index(default_metric) if default_metric in ["sum", "mean", "count", "max", "min"] else 0,
                                      key=f"kpi_metric_{i}")
            if st.session_state.kpi_count > 1:
                if st.button("🗑️", key=f"remove_kpi_{i}", help="Remove this KPI"):
                    st.session_state.kpi_count -= 1
                    st.rerun()
            confirmed_kpis.append({"label": label_inp, "column": col_sel, "metric": metric_sel})

    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        if st.button("➕ Add KPI", key="add_kpi") and st.session_state.kpi_count < 6:
            st.session_state.kpi_count += 1
            st.rerun()

    if st.button("🚀 Build Dashboard", use_container_width=True, key="build_dash"):
        st.session_state.build_dashboard = True

    if st.session_state.get("build_dashboard"):

        # helper to render a chart block
        def render_chart(block, height=320):
            clines = {l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip() for l in block.split("\n") if ":" in l}
            title     = clines.get("TITLE", "Chart")
            ctype     = clines.get("TYPE", "bar").lower()
            x_col     = clines.get("X", "").strip()
            y_col     = clines.get("Y", "").strip()
            color_col = clines.get("COLOR", "none").strip()

            # Case-insensitive column matching + extract from sum(col) format
            col_map = {c.lower(): c for c in df.columns}
            import re as _re
            def resolve_col(name):
                if not name: return name
                m = _re.match(r'\w+\((\w+)\)', name)
                if m: name = m.group(1)
                return col_map.get(name.lower(), name)

            x_col = resolve_col(x_col)
            y_col = resolve_col(y_col)
            color_col = resolve_col(color_col) if color_col and color_col.lower() != "none" else None

            # Validate color_col: must be categorical, not numeric, not same as x/y
            if color_col:
                if color_col not in df.columns:
                    color_col = None
                elif pd.api.types.is_numeric_dtype(df[color_col]):
                    color_col = None
                elif color_col == x_col or color_col == y_col:
                    color_col = None
                elif df[color_col].nunique() > 15:
                    color_col = None  # too many categories → legend becomes unreadable

            # fallback if y_col not found → use first numeric col
            if y_col not in df.columns and y_col != "count":
                num_fallback = [c for c in df.select_dtypes(include="number").columns if c != x_col]
                y_col = num_fallback[0] if num_fallback else "count"

            st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)

            PALETTE = ["#1FAA8A", "#7DE6B0", "#178F6F", "#7FD1A6", "#0B3A2F",
                       "#4ecba0", "#2d8f6f", "#a8f0d0", "#0d5c42", "#5de0b0"]
            SCALE   = ["#1FAA8A", "#178F6F", "#0B3A2F"]

            BASE_LAYOUT = dict(
                plot_bgcolor="#020504", paper_bgcolor="#020504",
                font=dict(color="#E6F1EC", size=12),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=40),
                height=height,
                legend=dict(
                    title=dict(font=dict(color="#7DE6B0", size=11)),
                    font=dict(color="#E6F1EC", size=11),
                    bgcolor="rgba(15,35,28,0.85)",
                    bordercolor="rgba(31,170,138,0.4)",
                    borderwidth=1,
                    orientation="v",
                    x=1.01, y=1,
                    xanchor="left",
                    yanchor="top",
                    itemsizing="constant",
                    tracegroupgap=4
                ),
                xaxis=dict(
                    tickfont=dict(color="#91c3a2", size=11),
                    gridcolor="rgba(31,170,138,0.08)",
                    linecolor="rgba(31,170,138,0.15)"
                ),
                yaxis=dict(
                    tickfont=dict(color="#91c3a2", size=11),
                    gridcolor="rgba(31,170,138,0.08)",
                    linecolor="rgba(31,170,138,0.15)"
                )
            )

            try:
                plot_df = df.copy()
                # Remove Unknown values from categorical columns
                for c in plot_df.select_dtypes(include="object").columns:
                    plot_df = plot_df[plot_df[c].astype(str).str.strip() != "Unknown"]

                # Convert date-like x columns to period strings
                if x_col in plot_df.columns:
                    try:
                        col_lower_x = x_col.lower()
                        date_hints = ["date", "time", "created", "updated", "birth", "timestamp"]
                        if any(k in col_lower_x for k in date_hints):
                            parsed = pd.to_datetime(plot_df[x_col], errors="coerce")
                            if parsed.notna().mean() > 0.7:
                                plot_df[x_col] = parsed.dt.to_period("M").astype(str)
                    except:
                        pass

                # Force line chart for year/date columns
                if x_col and "year" in x_col.lower():
                    ctype = "line"

                if ctype == "scatter":
                    ctype = "bar"
                if ctype == "histogram":
                    ctype = "bar"

                if ctype == "pie":
                    grp = plot_df[x_col].value_counts().head(7).reset_index()
                    grp.columns = [x_col, "count"]
                    fig = px.pie(
                        grp, names=x_col, values="count",
                        color_discrete_sequence=PALETTE, hole=0.4
                    )
                    fig.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        textfont=dict(color="#E6F1EC", size=11),
                        pull=[0.03] * len(grp)
                    )
                    fig.update_layout(**BASE_LAYOUT)
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            font=dict(color="#E6F1EC", size=11),
                            bgcolor="rgba(15,35,28,0.85)",
                            bordercolor="rgba(31,170,138,0.4)",
                            borderwidth=1,
                            title=dict(text=x_col, font=dict(color="#7DE6B0", size=11)),
                            itemsizing="constant"
                        )
                    )

                elif ctype == "scatter":
                    if x_col in plot_df.columns and y_col in plot_df.columns:
                        fig = px.scatter(
                            plot_df, x=x_col, y=y_col,
                            color=color_col,
                            color_discrete_sequence=PALETTE,
                            opacity=0.75,
                            labels={x_col: x_col, y_col: y_col, color_col: color_col} if color_col else None
                        )
                        fig.update_traces(marker=dict(size=7))
                        fig.update_layout(**BASE_LAYOUT)
                        fig.update_layout(
                            showlegend=color_col is not None,
                            legend=dict(title=dict(text=color_col or "", font=dict(color="#7DE6B0", size=11))) if color_col else {}
                        )
                    else:
                        raise ValueError(f"Columns not found: {x_col}, {y_col}")

                elif ctype == "histogram":
                    if x_col in plot_df.columns:
                        fig = px.histogram(
                            plot_df, x=x_col,
                            color=color_col,
                            color_discrete_sequence=PALETTE,
                            nbins=30,
                            labels={x_col: x_col, color_col: color_col} if color_col else None
                        )
                        fig.update_layout(**BASE_LAYOUT)
                        fig.update_layout(
                            showlegend=color_col is not None,
                            bargap=0.05,
                            legend=dict(title=dict(text=color_col or "", font=dict(color="#7DE6B0", size=11))) if color_col else {}
                        )
                    else:
                        raise ValueError(f"Column not found: {x_col}")

                elif ctype == "line":
                    if y_col == "count":
                        grp = plot_df[x_col].value_counts().sort_index().reset_index()
                        grp.columns = [x_col, "count"]
                        fig = px.line(grp, x=x_col, y="count",
                                      color_discrete_sequence=PALETTE,
                                      markers=True)
                        fig.update_layout(**BASE_LAYOUT)
                        fig.update_layout(showlegend=False)
                    else:
                        if color_col:
                            grp = plot_df.groupby([x_col, color_col])[y_col].sum().reset_index()
                            fig = px.line(grp, x=x_col, y=y_col, color=color_col,
                                          color_discrete_sequence=PALETTE, markers=True,
                                          labels={color_col: color_col})
                            fig.update_layout(**BASE_LAYOUT)
                            fig.update_layout(
                                showlegend=True,
                                legend=dict(title=dict(text=color_col, font=dict(color="#7DE6B0", size=11)))
                            )
                        else:
                            grp = plot_df.groupby(x_col)[y_col].sum().reset_index()
                            fig = px.line(grp, x=x_col, y=y_col,
                                          color_discrete_sequence=PALETTE, markers=True)
                            fig.update_layout(**BASE_LAYOUT)
                            fig.update_layout(showlegend=False)
                    fig.update_traces(line=dict(width=2.5))

                else:  # bar (default)
                    if y_col == "count":
                        grp = plot_df[x_col].value_counts().head(10).reset_index()
                        grp.columns = [x_col, "count"]
                        if color_col:
                            fig = px.bar(grp, x=x_col, y="count",
                                         color_discrete_sequence=PALETTE)
                        else:
                            fig = px.bar(grp, x=x_col, y="count",
                                         color="count", color_continuous_scale=SCALE)
                        fig.update_layout(**BASE_LAYOUT)
                        fig.update_layout(showlegend=False, bargap=0.15)
                    else:
                        if color_col:
                            grp = plot_df.groupby([x_col, color_col])[y_col].sum().reset_index()
                            grp = grp.sort_values(y_col, ascending=False)
                            # Limit to top 10 x values to keep chart readable
                            top_x = grp.groupby(x_col)[y_col].sum().nlargest(10).index
                            grp = grp[grp[x_col].isin(top_x)]
                            fig = px.bar(grp, x=x_col, y=y_col, color=color_col,
                                         color_discrete_sequence=PALETTE, barmode="group",
                                         labels={color_col: color_col})
                            fig.update_layout(**BASE_LAYOUT)
                            fig.update_layout(
                                showlegend=True,
                                bargap=0.15,
                                legend=dict(title=dict(text=color_col, font=dict(color="#7DE6B0", size=11)))
                            )
                        else:
                            grp = plot_df.groupby(x_col)[y_col].sum().reset_index()
                            grp = grp.sort_values(y_col, ascending=False).head(10)
                            fig = px.bar(grp, x=x_col, y=y_col,
                                         color=y_col, color_continuous_scale=SCALE)
                            fig.update_layout(**BASE_LAYOUT)
                            fig.update_layout(showlegend=False, bargap=0.15)

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.caption(f"Could not render chart: {e}")

        # ── KPI ROW ──────────────────────────────────────────
        n = len(confirmed_kpis)
        kpi_card_cols = st.columns(n)
        for i, kpi in enumerate(confirmed_kpis):
            col  = kpi["column"]
            metric = kpi["metric"]
            label  = kpi["label"]
            if col in df.columns:
                if metric == "sum":    val = df[col].sum()
                elif metric == "mean": val = round(df[col].mean(), 2)
                elif metric == "count": val = len(df[col].dropna())
                elif metric == "max":  val = df[col].max()
                elif metric == "min":  val = df[col].min()
                else: val = 0
                try:
                    val_fmt = f"{val:,.0f}" if abs(float(val)) >= 1 else f"{float(val):,.3f}"
                except:
                    val_fmt = str(val)
                kpi_card_cols[i].markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val_fmt}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # AI generates chart plan
        if "dashboard_charts" not in st.session_state or st.session_state.dashboard_charts is None:
            # Build a rich schema description so the AI picks meaningful charts
            col_info_lines = []
            for c in df.columns:
                dtype = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"
                n_unique = df[c].nunique()
                col_info_lines.append(f"  - {c} ({dtype}, {n_unique} unique values)")
            col_schema = "\n".join(col_info_lines)

            with st.spinner("AI is building your charts..."):
                st.session_state.dashboard_charts = ask_ai(
                    system_prompt=f"""You are a BI dashboard designer. Suggest exactly 4 professional, meaningful charts.

RULES:
- bar chart: use when X is categorical (≤ 20 unique values) and Y is a numeric column that is DIRECTLY related to X (e.g. sales by region, quantity by product). Do NOT compare unrelated columns.
- line chart: use ONLY when X is a date/time column. Y must be a numeric column that changes over time (e.g. revenue over time). Never use line for non-date X.
- pie chart: use ONLY when X is categorical with 3–7 unique values and you want to show proportions of a whole. Y must be 'count' or a numeric column summed per category.
- COLOR field: use a categorical column ONLY if it adds a meaningful third dimension (e.g. sales by region colored by product category). The color column must be logically related to both X and Y. Otherwise write 'none'.
- Do NOT use a numeric column as COLOR.
- Do NOT pair columns that have no logical business relationship.
- Each chart must tell one clear, specific business story using columns that are naturally related.
- Prefer charts that show the most impactful business metrics (revenue, quantity, performance, trends).

Return in this strict format:
TITLE: <descriptive chart title explaining what it shows>
TYPE: <bar|line|pie|scatter|histogram>
X: <column name>
Y: <numeric column name or 'count'>
COLOR: <categorical column name or 'none'>
---
Repeat 4 times. Only use columns that exist in the schema below. No extra text.

Available columns with types:
{col_schema}""",
                    user_prompt=f"Summary:\n{summary}\nSample:\n{sample_data}"
                )

        chart_blocks = [b.strip() for b in st.session_state.dashboard_charts.split("---") if b.strip()][:4]
        # ── ROW 1: big chart left (2/3) + tall chart right (1/3) ──
        if len(chart_blocks) >= 2:
            r1_left, r1_right = st.columns([2, 1])
            with r1_left:
                with st.container():
                    render_chart(chart_blocks[0], height=360)
            with r1_right:
                with st.container():
                    render_chart(chart_blocks[1], height=360)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 2: two equal charts ──
        if len(chart_blocks) >= 4:
            r2_left, r2_right = st.columns(2)
            with r2_left:
                render_chart(chart_blocks[2], height=300)
            with r2_right:
                render_chart(chart_blocks[3], height=300)

# =========================
# 🤖 ML CHAT PAGE
# =========================
elif st.session_state.page == "ml":
    import json as _json

    if st.button("← Back to Analysis"):
        st.session_state.page = "main"
        st.session_state.user_path = None
        for k in ["ml_chat", "ml_trained", "ml_leaderboard", "ml_target_col", "ml_data_type", "ml_task_type", "ml_step", "ml_feature_cols", "ml_ai_features_done"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.title("🤖 ML Studio")

    df = st.session_state.get("df_for_ml", None)
    if df is None:
        st.warning("No data found. Please go back and upload a file.")
        st.stop()

    num_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
    cat_cols = list(df.select_dtypes(include=["object"]).columns)
    all_cols = list(df.columns)

    if "ml_step" not in st.session_state:
        st.session_state.ml_step = "select_target"

    with st.expander("📋 Dataset Overview", expanded=True):
        st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
        st.dataframe(df.head(10), use_container_width=True)

    # ============================================================
    # STEP 1: Select Target → AI picks features → Review → Detect type
    # ============================================================
    if st.session_state.ml_step == "select_target":
        st.subheader("🎯 Step 1: Select Target & AI Feature Selection")
        st.caption("Choose the column to predict — AI will suggest the most useful features for it.")

        target_col = st.selectbox("Target Column (what to predict)", options=[""] + all_cols, key="ml_target_selector")

        if "ml_ai_features_done" not in st.session_state:
            st.session_state.ml_ai_features_done = False

        # Button to trigger AI feature suggestion
        if target_col and not st.session_state.ml_ai_features_done:
            if st.button("🤖 Auto-Select Useful Features", use_container_width=True):
                with st.spinner("🧠 AI is analyzing which features are useful for predicting this target..."):
                    col_info = "\n".join(
                        [f"- {c}: {'numeric' if c in num_cols else 'categorical'} "
                         f"({df[c].nunique()} unique, e.g. {str(df[c].dropna().iloc[0]) if len(df[c].dropna())>0 else 'N/A'})"
                         for c in all_cols if c != target_col]
                    )
                    sys = (
                        "You are an expert ML feature selector. Given a target column and a list of available columns, "
                        "select ONLY the columns that are useful features for predicting the target.\n\n"
                        "Rules:\n"
                        "- Exclude: ID columns (OrderID, ProductID, CustomerID, etc.), raw dates, and anything that leaks the target\n"
                        "- Include: columns that have predictive value (numeric measurements, categories like shipping mode, region, etc.)\n"
                        "- Return ONLY valid JSON array of column names, no markdown\n"
                        'Example: ["col1", "col2", "col3"]'
                    )
                    usr = f"Target: {target_col}\n\nAvailable columns:\n{col_info}"
                    raw = ask_ai(sys, usr)
                    raw_clean = re.sub(r"```json|```", "", raw).strip()
                    # Hard exclusion list — these are NEVER useful as features
                    _hard_exclude = {"id", "orderid", "productid", "customerid", "userid", "sku"}
                    try:
                        suggested = _json.loads(raw_clean)
                        suggested = [c for c in suggested if c in all_cols and c != target_col and c.lower().strip() not in _hard_exclude]
                    except:
                        suggested = [c for c in all_cols if c != target_col and c.lower().strip() not in _hard_exclude]
                    if len(suggested) < 1:
                        suggested = [c for c in all_cols if c != target_col and c.lower().strip() not in _hard_exclude]
                    st.session_state.ml_feature_cols = suggested
                    st.session_state.ml_ai_features_done = True
                    st.rerun()

        # Show multiselect with AI suggestions (user can modify)
        if st.session_state.ml_ai_features_done and target_col:
            remaining = [c for c in all_cols if c != target_col]
            feature_cols = st.multiselect(
                "📌 Suggested Features (you can add/remove):",
                options=remaining,
                default=st.session_state.ml_feature_cols,
                key="ml_feat_selector"
            )

            if len(feature_cols) >= 1:
                if st.button("🔍 Detect Data Type & Continue", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your data type..."):
                        col_info_full = "\n".join(
                            [f"- {c}: {'numeric' if c in num_cols else 'categorical'} "
                             f"({df[c].nunique()} unique, e.g. {str(df[c].dropna().iloc[0]) if len(df[c].dropna())>0 else 'N/A'})"
                             for c in all_cols]
                        )
                        sys = (
                            "You are an ML expert. Based on the dataset columns and the selected target column, "
                            "determine the learning type:\n"
                            "- SUPERVISED: target has clear labels/values for prediction\n"
                            "- UNSUPERVISED: no clear target or it's an identifier\n"
                            "- SEMISUPERVISED: only some rows have labels\n"
                            "Return ONLY the word: SUPERVISED, UNSUPERVISED, or SEMISUPERVISED."
                        )
                        usr = f"Target column: {target_col}\n\nColumns:\n{col_info_full}"
                        raw = ask_ai(sys, usr).strip().upper()
                        if "SEMI" in raw:
                            data_type = "SEMISUPERVISED"
                        elif "UNSUPERVISED" in raw:
                            data_type = "UNSUPERVISED"
                        else:
                            data_type = "SUPERVISED"

                    st.session_state.ml_target_col = target_col
                    st.session_state.ml_feature_cols = feature_cols
                    st.session_state.ml_data_type = data_type
                    st.session_state.ml_step = "task_selection"
                    st.rerun()
            else:
                st.warning("Please select at least one feature column.")

        # Reset button
        if st.session_state.ml_ai_features_done and st.button("🔄 Try a different target"):
            st.session_state.ml_ai_features_done = False
            st.session_state.pop("ml_feature_cols", None)
            st.rerun()

    # ============================================================
    # STEP 2: Show DATA TYPE + Choose Task
    # ============================================================
    elif st.session_state.ml_step == "task_selection":
        data_type = st.session_state.ml_data_type
        target_col = st.session_state.ml_target_col

        type_colors = {"SUPERVISED": "#1FAA8A", "UNSUPERVISED": "#FFA500", "SEMISUPERVISED": "#9B59B6"}
        color = type_colors.get(data_type, "#1FAA8A")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(31,170,138,0.12), rgba(11,58,47,0.25));
            border: 2px solid {color};
            border-radius: 14px; padding: 20px 24px; margin: 12px 0; text-align: center;
        ">
            <div style="color: #888; font-size: 0.8rem; letter-spacing: 2px;">DATA TYPE</div>
            <div style="color: {color}; font-size: 2.2rem; font-weight: 800;">{data_type}</div>
            <div style="color: #c8e6d4; font-size: 0.95rem; margin-top: 6px;">Target: <code>{target_col}</code></div>
        </div>
        """, unsafe_allow_html=True)

        if data_type == "SUPERVISED":
            st.subheader("📌 Step 2: Choose Task Type")
            task_type = st.radio(
                "Select task:",
                ["classification", "regression"],
                format_func=lambda x: "🏷️ Classification" if x == "classification" else "📈 Regression",
                horizontal=True,
            )
            if st.button("🚀 Train 3 Models", use_container_width=True):
                st.session_state.ml_task_type = task_type
                st.session_state.ml_step = "training"
                st.rerun()
        else:
            st.warning(f"⚠️ {data_type} learning is not supported yet. This tool focuses on SUPERVISED learning.")
            if st.button("◀️ Try Again"):
                st.session_state.ml_step = "select_target"
                st.rerun()

    # ============================================================
    # STEP 3: Train 3 Models & Show Best
    # ============================================================
    elif st.session_state.ml_step == "training":
        target_col = st.session_state.ml_target_col
        feature_cols = st.session_state.ml_feature_cols
        task_type = st.session_state.ml_task_type
        is_regression = (task_type == "regression")

        # Safety filter: remove any ID-like columns that slipped through
        _hard_exclude = {"id", "orderid", "productid", "customerid", "userid", "sku"}
        feature_cols = [c for c in feature_cols if c.lower().strip() not in _hard_exclude]

        if len(feature_cols) < 1:
            st.error("❌ No feature columns selected.")
            st.stop()

        with st.spinner(f"Training 3 {task_type} models..."):
            ml_df = df[feature_cols + [target_col]].dropna().copy()
            if len(ml_df) < 30:
                st.error(f"❌ Not enough data ({len(ml_df)} rows). Need at least 30.")
                st.stop()

            X = ml_df[feature_cols].copy()
            y = ml_df[target_col].copy()

            cats_in_feats = [c for c in feature_cols if c in cat_cols]
            if cats_in_feats:
                X = pd.get_dummies(X, columns=cats_in_feats, drop_first=True)

            le = None
            if not is_regression:
                if y.dtype == object:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                else:
                    y = y.astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if is_regression:
                model_configs = [
                    ("Linear Regression", LinearRegression(), {"fit_intercept": [True]}),
                    ("Random Forest", RandomForestRegressor(random_state=42, n_jobs=-1),
                     {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}),
                    ("XGBoost", XGBRegressor(random_state=42, verbosity=0),
                     {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}),
                ]
                scoring = "r2"
                sort_col = "R²"
            else:
                model_configs = [
                    ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
                     {"C": [0.1, 1, 10]}),
                    ("Random Forest", RandomForestClassifier(random_state=42, n_jobs=-1),
                     {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}),
                    ("XGBoost", XGBClassifier(random_state=42, verbosity=0),
                     {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}),
                ]
                scoring = "accuracy"
                sort_col = "Accuracy"

            leaderboard = []
            best_model, best_score, best_name = None, -np.inf, ""

            progress_bar = st.progress(0)
            for i, (name, model, param_grid) in enumerate(model_configs):
                gs = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=0)
                gs.fit(X_train_s, y_train)
                best = gs.best_estimator_
                y_pred = best.predict(X_test_s)

                if is_regression:
                    score = r2_score(y_test, y_pred)
                    leaderboard.append({
                        "Model": name, "R²": round(score, 4),
                        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
                        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 2),
                        "Best Params": str(gs.best_params_),
                    })
                else:
                    score = accuracy_score(y_test, y_pred)
                    leaderboard.append({
                        "Model": name,
                        "Accuracy": round(score, 4),
                        "Precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                        "Recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                        "F1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                        "Best Params": str(gs.best_params_),
                    })

                if score > best_score:
                    best_score, best_model, best_name = score, best, name
                progress_bar.progress((i + 1) / len(model_configs))

            progress_bar.empty()

            st.session_state.ml_trained = {
                "model": best_model, "name": best_name,
                "scaler": scaler, "feats": list(X.columns),
                "cats": cats_in_feats, "orig_feats": feature_cols,
                "is_reg": is_regression, "le": le,
                "target": target_col, "sort_col": sort_col,
                "best_score": best_score,
            }
            st.session_state.ml_leaderboard = leaderboard
            st.session_state.ml_step = "results"
            st.rerun()

    # ============================================================
    # STEP 4: Show Results + Prediction
    # ============================================================
    elif st.session_state.ml_step == "results":
        target_col = st.session_state.ml_target_col
        task_type = st.session_state.ml_task_type
        ml_trained = st.session_state.ml_trained
        leaderboard = st.session_state.ml_leaderboard
        is_regression = ml_trained["is_reg"]
        best_name = ml_trained["name"]
        best_score = ml_trained["best_score"]
        sort_col = ml_trained["sort_col"]

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:16px;">
            <span style="background:#1FAA8A; padding:3px 14px; border-radius:20px; color:white; font-weight:600; font-size:0.8rem;">SUPERVISED</span>
            <span style="background:rgba(31,170,138,0.15); padding:3px 14px; border-radius:20px; color:#c8e6d4; font-size:0.8rem;">
                {'📈 Regression' if is_regression else '🏷️ Classification'}
            </span>
            <span style="background:rgba(31,170,138,0.15); padding:3px 14px; border-radius:20px; color:#c8e6d4; font-size:0.8rem;">
                🎯 {target_col}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("🔮 Make a Prediction")
        st.caption("Enter values for each feature to predict:")

        input_vals = {}
        pred_cols = st.columns(3)
        for i, feat in enumerate(ml_trained["orig_feats"]):
            with pred_cols[i % 3]:
                sample_val = df[feat].dropna().iloc[0] if len(df[feat].dropna()) > 0 else 0
                if feat in num_cols:
                    input_vals[feat] = st.number_input(
                        feat,
                        value=float(sample_val) if isinstance(sample_val, (int, float)) else 0.0,
                        key=f"pred_{feat}"
                    )
                else:
                    unique_vals = df[feat].dropna().unique().tolist()
                    input_vals[feat] = st.selectbox(feat, options=unique_vals, key=f"pred_{feat}")

        if st.button("🔮 Predict", use_container_width=True, type="primary"):
            try:
                inp = pd.DataFrame([input_vals])
                if ml_trained["cats"]:
                    inp = pd.get_dummies(inp, columns=ml_trained["cats"], drop_first=True)
                for c in ml_trained["feats"]:
                    if c not in inp.columns:
                        inp[c] = 0
                inp = inp[ml_trained["feats"]]
                pred = ml_trained["model"].predict(ml_trained["scaler"].transform(inp))

                if is_regression:
                    pred_label = f"**{pred[0]:,.2f}**"
                elif ml_trained["le"] is not None:
                    pred_label = f"**{ml_trained['le'].inverse_transform(pred.astype(int))[0]}**"
                else:
                    pred_label = f"**{int(pred[0])}**"

                st.success(f"🔮 Prediction for `{target_col}`: {pred_label}")
                st.caption("⚠️ AI Data Analysis Tool can make mistakes.")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

        st.divider()
        if st.button("◀️ Start Over"):
            for k in ["ml_target_col", "ml_feature_cols", "ml_trained", "ml_leaderboard", "ml_data_type", "ml_task_type", "ml_step", "ml_ai_features_done"]:
                st.session_state.pop(k, None)
            st.rerun()
