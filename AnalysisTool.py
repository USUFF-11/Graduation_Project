import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

# =========================
# 🔐 OPENROUTER CONFIG
# =========================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="API_KEY_HERE"
)

def ask_ai(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="AI Data Tool", layout="wide")

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
def clean_dataframe(df):
    cleaning_report = []

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip().str.capitalize().str.replace(" ", "_")

    date_keywords = ["date", "time", "day", "month", "year", "birth", "created", "updated", "timestamp", "dt"]
    numeric_keywords = ["price", "cost", "salary", "amount", "revenue", "profit", "fee", "rate", "score",
                        "count", "qty", "quantity", "total", "weight", "height", "age", "distance", "duration"]
    bool_keywords = ["is_", "has_", "flag", "active", "enabled", "returned", "damaged"]

    # ⚡ Use sample for type detection on large files
    sample = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

    for col in df.columns:
        col_lower = col.lower()
        if df[col].dtype != object:
            continue

        if any(k in col_lower for k in date_keywords):
            try:
                converted = pd.to_datetime(sample[col], infer_datetime_format=True, errors="coerce")
                if converted.notna().mean() > 0.7:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce").dt.date
                    cleaning_report.append(f"📅 `{col}`: converted to **date**.")
                    continue
            except:
                pass

        if any(k in col_lower for k in numeric_keywords):
            try:
                converted = pd.to_numeric(sample[col].str.replace(",", "").str.replace("$", "").str.strip(), errors="coerce")
                if converted.notna().mean() > 0.7:
                    df[col] = pd.to_numeric(df[col].str.replace(",", "").str.replace("$", "").str.strip(), errors="coerce")
                    cleaning_report.append(f"🔢 `{col}`: converted to **numeric**.")
                    continue
            except:
                pass

        if any(k in col_lower for k in bool_keywords):
            try:
                bool_map = {"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False}
                if sample[col].str.lower().isin(bool_map.keys()).mean() > 0.7:
                    df[col] = df[col].str.lower().map(bool_map)
                    cleaning_report.append(f"✅ `{col}`: converted to **boolean**.")
                    continue
            except:
                pass

        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except:
            pass

    duplicates_count = df.duplicated().sum()
    id_keywords = ["id", "code", "key", "uid", "ref", "sku"]
    id_candidates = [c for c in df.columns if any(k == c.lower() or c.lower().endswith(k) or c.lower().startswith(k) for k in id_keywords)]
    id_col = max(id_candidates, key=lambda c: df[c].nunique()) if id_candidates else None
    df = df.drop_duplicates()
    if duplicates_count > 0:
        if id_col:
            cleaning_report.append(f"🗑️ Removed **{duplicates_count}** duplicate rows based on `{id_col}`.")
        else:
            cleaning_report.append(f"🗑️ Removed **{duplicates_count}** duplicate rows.")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    total_rows = len(df)
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue
        pct = missing / total_rows
        if pct > 0.3:
            df.drop(columns=[col], inplace=True)
            cleaning_report.append(f"🗑️ `{col}`: dropped — **{pct*100:.0f}%** missing values (too many).")
        elif pct > 0.05:
            df[col].fillna(df[col].mean(), inplace=True)
            cleaning_report.append(f"🔢 `{col}`: filled **{missing}** missing values with mean ({pct*100:.0f}% missing).")
        else:
            df[col].fillna(df[col].median(), inplace=True)
            cleaning_report.append(f"🔢 `{col}`: filled **{missing}** missing values with median.")

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Text consistency: strip + title case + normalize common variants
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)  # collapse multiple spaces

        # Normalize common country/value variants
        replacements = {
            r'^u\.?s\.?a?\.?$': 'USA', r'^united states.*': 'USA',
            r'^u\.?k\.?$': 'UK', r'^united kingdom.*': 'UK',
            r'^nan$': None, r'^none$': None, r'^n/a$': None, r'^-$': None
        }
        for pattern, val in replacements.items():
            mask = df[col].str.lower().str.match(pattern, na=False)
            if mask.any():
                df.loc[mask, col] = val

        missing = df[col].isnull().sum()
        if missing > 0:
            pct = missing / total_rows
            if pct > 0.3:
                df.drop(columns=[col], inplace=True)
                cleaning_report.append(f"🗑️ `{col}`: dropped — **{pct*100:.0f}%** missing values.")
            else:
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                df[col].fillna(fill_val, inplace=True)
                cleaning_report.append(f"🔤 `{col}`: filled **{missing}** missing values with most common value.")

    return df, cleaning_report

@st.cache_data(show_spinner=False)
def get_context(df):
    summary = df.describe().round(2).to_string()
    columns = ", ".join(df.columns)
    sample_data = df.sample(min(20, len(df))).to_string()
    return summary, columns, sample_data

# =========================
# 📄 PAGE ROUTING
# =========================
if "page" not in st.session_state:
    st.session_state.page = "main"

# Splash screen on first load
if "app_loaded" not in st.session_state:
    st.markdown("""
    <style>
    #splash {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: #080b14; display: flex; flex-direction: column;
        align-items: center; justify-content: center; z-index: 9999;
        animation: fadeOut 0.6s ease 1.8s forwards;
    }
    @keyframes fadeOut { to { opacity: 0; pointer-events: none; } }
    .splash-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #7c83fd, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 12px; letter-spacing: -1px;
    }
    .splash-sub { color: #4a5180; font-size: 1rem; margin-bottom: 40px; }
    .splash-bar {
        width: 220px; height: 3px; background: #1a1f35; border-radius: 99px; overflow: hidden;
    }
    .splash-fill {
        height: 100%; width: 0%;
        background: linear-gradient(90deg, #7c83fd, #a78bfa);
        border-radius: 99px;
        animation: fill 1.8s ease forwards;
    }
    @keyframes fill { to { width: 100%; } }
    </style>
    <div id="splash">
        <div class="splash-title">✦ AI Data Analysis Tool</div>
        <div class="splash-sub">Powered by OpenRouter · Built with Streamlit</div>
        <div class="splash-bar"><div class="splash-fill"></div></div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.app_loaded = True

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    /* ── Background ── */
    .stApp { background-color: #080b14; color: #e2e8f0; }
    .block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1400px !important; }

    /* ── Hide Streamlit default top bar loader ── */
    div[data-testid="stToolbar"] { display: none !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #7c83fd, #a78bfa) !important; }

    /* ── Title ── */
    h1 {
        background: linear-gradient(135deg, #7c83fd 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem !important;
    }

    /* ── Subheaders ── */
    h2, h3 {
        color: #c4c9ff !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        letter-spacing: -0.2px;
        border-bottom: 1px solid #1e2235;
        padding-bottom: 8px;
        margin-top: 2rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a1f35 0%, #1e2440 100%);
        color: #a5b4fc;
        border: 1px solid #2d3555;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 0.88rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.2px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c83fd 0%, #a78bfa 100%);
        color: white;
        border-color: transparent;
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(124, 131, 253, 0.35);
    }
    .stButton > button:active { transform: translateY(0px); }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        background-color: #111627;
        color: #e2e8f0;
        border: 1px solid #2d3555;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.9rem;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c83fd !important;
        box-shadow: 0 0 0 3px rgba(124,131,253,0.15) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #111627 !important;
        border: 1px solid #2d3555 !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    /* ── File uploader ── */
    .stFileUploader {
        background: linear-gradient(135deg, #111627 0%, #131929 100%);
        border: 2px dashed #2d3555;
        border-radius: 16px;
        padding: 12px;
        transition: border-color 0.2s;
    }
    .stFileUploader:hover { border-color: #7c83fd; }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #1e2235;
    }

    /* ── Alerts ── */
    .stSuccess {
        background: linear-gradient(135deg, #0d2218 0%, #0f2a1c 100%) !important;
        border-left: 3px solid #22c55e !important;
        border-radius: 10px !important;
    }
    .stInfo {
        background: linear-gradient(135deg, #0d1a2e 0%, #0f1f38 100%) !important;
        border-left: 3px solid #3b82f6 !important;
        border-radius: 10px !important;
    }
    .stWarning {
        background: linear-gradient(135deg, #1e1608 0%, #251c0a 100%) !important;
        border-left: 3px solid #f59e0b !important;
        border-radius: 10px !important;
    }
    .stError {
        background: linear-gradient(135deg, #1e0d0d 0%, #250f0f 100%) !important;
        border-left: 3px solid #ef4444 !important;
        border-radius: 10px !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c83fd, #a78bfa) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div {
        background-color: #1e2235 !important;
        border-radius: 99px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #7c83fd !important; }

    /* ── Divider ── */
    hr { border-color: #1e2235 !important; margin: 1.5rem 0 !important; }

    /* ── Slider ── */
    .stSlider > div > div > div { background: #7c83fd !important; }

    /* ── Caption ── */
    .stCaption { color: #64748b !important; font-size: 0.8rem !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2d3555; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: #7c83fd; }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom: 0.5rem;'>
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>
        <span style='font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#7c83fd,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>✦ AI Data Analysis Tool</span>
    </div>
    <p style='color:#64748b;font-size:0.9rem;margin:0'>Upload your CSV and get instant insights, predictions, and dashboards — powered by AI.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your data", type=["csv"])

if uploaded_file:
    # مسح الـ cache لو الملف اتغير
    prev_name = st.session_state.get("last_file_name", None)
    prev_size = st.session_state.get("last_file_size", None)
    if prev_name != uploaded_file.name or prev_size != uploaded_file.size:
        keys_to_clear = ["auto_insights", "copilot_response", "questions_list",
                         "dashboard_kpis", "dashboard_charts", "build_dashboard",
                         "ml_option", "user_path", "df_for_dashboard", "selected_question"]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.session_state.last_file_name = uploaded_file.name
        st.session_state.last_file_size = uploaded_file.size
    st.session_state.uploaded_file = uploaded_file

if st.session_state.page == "main" and "uploaded_file" in st.session_state and st.session_state.uploaded_file:
    uploaded_file = st.session_state.uploaded_file

    file_size_mb = uploaded_file.size / (1024 * 1024)

    # ⚡ Progress bar
    progress = st.progress(0, text="📂 Loading file...")
    status = st.empty()

    # Step 1: Load
    status.info(f"📂 Loading file ({file_size_mb:.0f} MB)... please wait")
    file_bytes = uploaded_file.getvalue()
    raw_df = load_file(file_bytes, uploaded_file.name)
    progress.progress(40, text="✅ File loaded!")

    # Step 2: Clean
    status.info("🧹 Cleaning data... detecting types and fixing missing values")
    df, cleaning_report = clean_dataframe(raw_df)
    progress.progress(80, text="✅ Data cleaned!")

    # Step 3: Done
    status.success(f"✅ Ready! {len(df):,} rows × {len(df.columns)} columns loaded successfully.")
    progress.progress(100, text="✅ Done!")
    progress.empty()

    # =========================
    # 📊 SHOW DATA
    # =========================

    st.subheader("📊 Cleaned Data")
    st.caption(f"{len(df)} rows × {len(df.columns)} columns — showing first 500 rows")
    st.dataframe(df.head(500), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Cleaned Data", data=csv, file_name="cleaned_data.csv", mime="text/csv")

    st.subheader("🧹 Cleaning Report")
    if cleaning_report:
        for item in cleaning_report:
            st.markdown(item)
    else:
        st.success("✅ Data is clean! No issues found.")

    # =========================
    # 🛤️ PATH SELECTION
    # =========================

    st.divider()
    st.subheader("🛤️ What would you like to do next?")

    if "user_path" not in st.session_state:
        st.session_state.user_path = None

    path_col1, path_col2 = st.columns(2)
    with path_col1:
        if st.button("📊 Quick Dashboard\nBuild a dashboard directly from your data", key="path_dashboard", use_container_width=True):
            st.session_state.user_path = "dashboard"
    with path_col2:
        if st.button("🤖 Analyze & Build Dashboard\nRun ML analysis ", key="path_ml", use_container_width=True):
            st.session_state.user_path = "ml"

    st.divider()

    if st.session_state.user_path == "dashboard":
        st.session_state.df_for_dashboard = df
        st.session_state.page = "dashboard"
        st.rerun()

    if st.session_state.user_path not in ["ml", None]:
        st.stop()

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

    if st.session_state.user_path != "ml":
        st.stop()  # ← انتظر لحد ما المستخدم يضغط ML

    # =========================
    # 🔍 AUTO INSIGHTS
    # =========================
    st.subheader("🔍 Insights & Actions")
    if "auto_insights" not in st.session_state:
        with st.spinner("Analyzing your data..."):
            st.session_state.auto_insights = ask_ai(
                system_prompt="""You are a senior business analyst. Given a dataset, return exactly 4 insights in this strict format:
INSIGHT: <one sentence observation about the data>
ACTION: <one sentence business recommendation>
CHART: <one of: bar|line|pie — pick the best chart type for this insight>
CHART_COL: <exact column name from the dataset most relevant to this insight>
---
Repeat 4 times. The FIRST block must be the single most critical insight. No extra text.""",
                user_prompt=f"Columns: {columns}\n\nSummary:\n{summary}\n\nSample:\n{sample_data}"
            )

    # Parse insights
    raw = st.session_state.auto_insights
    blocks = [b.strip() for b in raw.split("---") if b.strip()]

    for idx, block in enumerate(blocks[:4]):
        lines = {l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip() for l in block.split("\n") if ":" in l}
        insight    = lines.get("INSIGHT", "")
        action     = lines.get("ACTION", "")
        chart_type = lines.get("CHART", "bar").lower()
        chart_col  = lines.get("CHART_COL", "").strip()

        if idx == 0:
            st.markdown("""<div style='background:linear-gradient(90deg,#2a1f5e,#1e2130);border:1px solid #7c83fd;border-radius:12px;padding:16px 20px;margin-bottom:8px'>
            <span style='color:#f5c542;font-size:1rem;font-weight:700'>⭐ Most Important Insight</span></div>""", unsafe_allow_html=True)
            col_text, col_chart_col = st.columns([1, 1.2])
            with col_text:
                st.warning(f"💡 {insight}")
                st.success(f"✅ {action}")
        else:
            col_text, col_chart_col = st.columns([1, 1.2])
            with col_text:
                st.info(f"💡 {insight}")
                st.success(f"✅ {action}")

        with col_chart_col:
            if chart_col in df.columns:
                try:
                    col_data = df[chart_col].dropna()
                    col_data = col_data[col_data.astype(str) != "Unknown"]
                    plot_df = df[df[chart_col].astype(str) != "Unknown"].copy()
                    is_numeric = pd.api.types.is_numeric_dtype(col_data)
                    n_unique = col_data.nunique()
                    col_lower = chart_col.lower()

                    # Detect date columns
                    is_date = False
                    if not is_numeric:
                        try:
                            parsed = pd.to_datetime(col_data, infer_datetime_format=True, errors="coerce")
                            if parsed.notna().mean() > 0.7:
                                is_date = True
                                col_data = parsed
                        except:
                            pass

                    if is_date:
                        # Time series → group by month/year and sum or count
                        date_df = df.copy()
                        date_df[chart_col] = pd.to_datetime(date_df[chart_col], errors="coerce")
                        date_df = date_df.dropna(subset=[chart_col])
                        date_df["_period"] = date_df[chart_col].dt.to_period("M").astype(str)
                        num_cols_avail = [c for c in date_df.select_dtypes(include="number").columns]
                        if num_cols_avail:
                            y_col = num_cols_avail[0]
                            ts = date_df.groupby("_period")[y_col].sum().reset_index()
                            ts.columns = ["Date", y_col]
                            fig = px.line(ts, x="Date", y=y_col, color_discrete_sequence=["#7c83fd"])
                        else:
                            ts = date_df["_period"].value_counts().sort_index().reset_index()
                            ts.columns = ["Date", "count"]
                            fig = px.line(ts, x="Date", y="count", color_discrete_sequence=["#7c83fd"])

                    elif is_numeric and n_unique > 20:
                        if any(k in col_lower for k in ["score", "rate", "avg", "satisfaction", "rating"]):
                            agg_label = "Average"
                            agg_val = round(col_data.mean(), 2)
                            fig = px.bar(pd.DataFrame({chart_col: [chart_col], agg_label: [agg_val]}),
                                         x=chart_col, y=agg_label, color_discrete_sequence=["#7c83fd"])
                        else:
                            cat_cols_avail = [c for c in plot_df.select_dtypes(include="object").columns]
                            if cat_cols_avail:
                                grp_col = cat_cols_avail[0]
                                grp = plot_df[plot_df[grp_col].astype(str) != "Unknown"].groupby(grp_col)[chart_col].sum().reset_index().sort_values(chart_col, ascending=False).head(8)
                                fig = px.bar(grp, x=grp_col, y=chart_col,
                                             color=chart_col, color_continuous_scale="Blues")
                            else:
                                fig = px.histogram(col_data, x=chart_col, color_discrete_sequence=["#7c83fd"])

                    elif is_numeric and n_unique <= 20:
                        grp = col_data.value_counts().sort_index().reset_index()
                        grp.columns = [chart_col, "count"]
                        fig = px.bar(grp, x=chart_col, y="count",
                                     color="count", color_continuous_scale="Blues",
                                     labels={"count": "Number of Records"})
                    else:
                        if chart_type == "pie":
                            grp = col_data.value_counts().head(6).reset_index()
                            grp.columns = [chart_col, "count"]
                            fig = px.pie(grp, names=chart_col, values="count",
                                         color_discrete_sequence=px.colors.sequential.Purples_r)
                        else:
                            grp = col_data.value_counts().head(8).reset_index()
                            grp.columns = [chart_col, "count"]
                            fig = px.bar(grp, x=chart_col, y="count",
                                         color="count", color_continuous_scale="Blues")

                    fig.update_layout(
                        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                        font_color="#e0e0e0", coloraxis_showscale=False,
                        margin=dict(l=10, r=10, t=10, b=10), height=220
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass
        st.divider()

    # =========================
    # 🤖 AI COPILOT
    # =========================

    st.subheader("🤖 AI Copilot")
    st.caption("Your AI analyst — detects problems, suggests solutions, and ranks priorities.")

    if "copilot_response" not in st.session_state:
        with st.spinner("AI Copilot is reviewing your data..."):
            st.session_state.copilot_response = ask_ai(
                system_prompt="""You are an AI business copilot. Analyze the dataset and return exactly this structure:

PROBLEM: <the biggest problem or risk you detect in the data>
SOLUTION: <specific action to fix it>
PRIORITY_1: <most important thing to focus on right now>
PRIORITY_2: <second most important thing>
PRIORITY_3: <third most important thing>

Be specific, use actual column names and numbers from the data. No extra text.""",
                user_prompt=f"Columns: {columns}\n\nSummary:\n{summary}\n\nSample:\n{sample_data}"
            )

    cop_lines = {l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip()
                 for l in st.session_state.copilot_response.split("\n") if ":" in l}

    c1, c2 = st.columns(2)
    with c1:
        st.error(f"🚨 Problem detected: {cop_lines.get('PROBLEM', '')}")
        st.success(f"💊 Suggested fix: {cop_lines.get('SOLUTION', '')}")
    with c2:
        st.markdown("**📋 Ranked Priorities:**")
        for rank, key in enumerate(["PRIORITY_1", "PRIORITY_2", "PRIORITY_3"], 1):
            val = cop_lines.get(key, "")
            if val:
                st.markdown(f"`#{rank}` {val}")

    st.divider()

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

    if question and st.session_state.ml_option is None:
        with st.spinner("Analyzing..."):
            answer = ask_ai(
                system_prompt="You are a data analyst. The user gave you a dataset. Answer the question directly using the actual data provided. Give a specific number or fact as the answer first, then a brief explanation if needed. Do NOT say you don't have access to the full dataset. Do NOT suggest how to find the answer. Just answer it.",
                user_prompt=context + "\n\nQuestion: " + question
            )
        st.subheader("🤖 Answer")
        st.write(answer)

    # =========================
    # 🤖 MACHINE LEARNING
    # =========================

    st.subheader("🤖 Smart Predictions & Analysis")

    ml_options = {
        "Predict a Number": "Forecast a numeric value like price or cost",
        "Predict a Category": "Classify records into categories like status or type",
        "Group Similar Records": "Cluster your data into meaningful segments",
        "Find Unusual Records": "Detect anomalies and outliers in your data"
    }

    if "ml_option" not in st.session_state:
        st.session_state.ml_option = None

    cols_ml = st.columns(4)
    for i, (label, desc) in enumerate(ml_options.items()):
        with cols_ml[i]:
            if st.button(label, key=f"ml_btn_{i}", use_container_width=True):
                st.session_state.ml_option = label
                st.session_state.selected_question = None
                st.rerun()
            if st.session_state.ml_option == label:
                st.markdown(f"<div style='text-align:center;font-size:0.72rem;color:#7c83fd;margin-top:-10px'>▲ selected</div>", unsafe_allow_html=True)

    ml_option = st.session_state.ml_option

    numeric_cols_list = list(df.select_dtypes(include=['int64', 'float64']).columns)
    all_cols_list = list(df.columns)

    if ml_option == "Predict a Number":
        st.markdown("Pick the column you want to predict (e.g. price, cost, quantity):")
        target = st.selectbox("Column to predict", numeric_cols_list, key="reg_target")
        features = [c for c in numeric_cols_list if c != target]
        if st.button("Run", key="run_reg") and features:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            st.success(f"Average prediction error: **{mae:.2f}** | Accuracy score: **{r2*100:.1f}%**")
            importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(5)
            col_chart, col_ai = st.columns([1.2, 1])
            with col_chart:
                st.markdown("**Top factors affecting the prediction:**")
                imp_df = importance.reset_index().rename(columns={"index": "Feature", 0: "Importance"})
                imp_df["Importance"] = (imp_df["Importance"] * 100).round(1)
                fig = px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    text=imp_df["Importance"].apply(lambda x: f"{x}%"),
                    color="Importance", color_continuous_scale="Blues"
                )
                fig.update_traces(textposition="outside", hovertemplate="<b>%{y}</b><br>Importance: %{x}%<extra></extra>")
                fig.update_layout(
                    plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                    font_color="#e0e0e0", coloraxis_showscale=False,
                    xaxis=dict(ticksuffix="%"),
                    yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_ai:
                with st.spinner("AI is analyzing the results..."):
                    ai_text = ask_ai(
                        system_prompt="You are a business analyst. Reply in this exact format:\nRESULT: <one sentence on model accuracy>\nWHY: <one sentence explaining why the top feature matters most>\nACTION: <one concrete business action to take>",
                        user_prompt=f"Model predicts '{target}'. MAE: {mae:.2f}, R2: {r2*100:.1f}%. Top feature: {importance.index[0]} ({importance.iloc[0]*100:.1f}%). All features: {importance.index.tolist()}."
                    )
                resp_lines = {l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip() for l in ai_text.split("\n") if ":" in l}
                st.info(f"📊 {resp_lines.get('RESULT', '')}")
                st.warning(f"🔍 {resp_lines.get('WHY', '')}")
                st.success(f"✅ {resp_lines.get('ACTION', '')}")

    elif ml_option == "Predict a Category":
        cat_cols_list = list(df.select_dtypes(include=['object']).columns)
        st.markdown("Pick the column you want to predict (e.g. status, category):")
        target = st.selectbox("Column to predict", cat_cols_list, key="clf_target")
        if st.button("Run", key="run_clf"):
            df_ml = df.copy()

            # Feature Engineering
            num_cols = list(df_ml.select_dtypes(include=['int64', 'float64']).columns)
            if "Unitprice" in df_ml.columns and "Unitcost" in df_ml.columns:
                df_ml["Profit_margin"] = df_ml["Unitprice"] - df_ml["Unitcost"]
            if "Unitsordered" in df_ml.columns and "Unitprice" in df_ml.columns:
                df_ml["Revenue"] = df_ml["Unitsordered"] * df_ml["Unitprice"]

            # Label Encode categorical columns (except target)
            other_cats = [c for c in df_ml.select_dtypes(include=['object']).columns if c != target]
            for col in other_cats:
                le = LabelEncoder()
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))

            features = list(df_ml.select_dtypes(include=['int64', 'float64']).columns)
            features = [f for f in features if f != target]

            X = df_ml[features].dropna()
            le_target = LabelEncoder()
            y = le_target.fit_transform(df_ml.loc[X.index, target].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Try GradientBoosting vs RandomForest, pick best
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
            rf.fit(X_train, y_train)
            gb.fit(X_train, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test))
            gb_acc = accuracy_score(y_test, gb.predict(X_test))

            if gb_acc >= rf_acc:
                model = gb
                acc = gb_acc
                model_name = "Gradient Boosting"
            else:
                model = rf
                acc = rf_acc
                model_name = "Random Forest"

            st.success(f"Prediction accuracy: **{acc*100:.1f}%** (using {model_name})")
            importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(5)
            col_chart, col_ai = st.columns([1.2, 1])
            with col_chart:
                st.markdown("**Top factors affecting the prediction:**")
                imp_df = importance.reset_index().rename(columns={"index": "Feature", 0: "Importance"})
                imp_df["Importance"] = (imp_df["Importance"] * 100).round(1)
                fig = px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    text=imp_df["Importance"].apply(lambda x: f"{x}%"),
                    color="Importance", color_continuous_scale="Purples"
                )
                fig.update_traces(textposition="outside", hovertemplate="<b>%{y}</b><br>Importance: %{x}%<extra></extra>")
                fig.update_layout(
                    plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                    font_color="#e0e0e0", coloraxis_showscale=False,
                    xaxis=dict(ticksuffix="%"),
                    yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_ai:
                with st.spinner("AI is analyzing the results..."):
                    ai_text = ask_ai(
                        system_prompt="You are a business analyst. Reply in this exact format:\nRESULT: <one sentence on model accuracy>\nWHY: <one sentence explaining why the top feature matters most>\nACTION: <one concrete business action to take>",
                        user_prompt=f"Model predicts '{target}'. Accuracy: {acc*100:.1f}%. Top feature: {importance.index[0]} ({importance.iloc[0]*100:.1f}%). All features: {importance.index.tolist()}."
                    )
                resp_lines = {l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip() for l in ai_text.split("\n") if ":" in l}
                st.info(f"📊 {resp_lines.get('RESULT', '')}")
                st.warning(f"🔍 {resp_lines.get('WHY', '')}")
                st.success(f"✅ {resp_lines.get('ACTION', '')}")

    elif ml_option == "Group Similar Records":
        st.markdown("How many groups do you want to split your data into?")
        n_clusters = st.slider("Number of groups", 2, 8, 3, key="n_clusters")
        if st.button("Run", key="run_cluster") and numeric_cols_list:
            X = df[numeric_cols_list].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df.loc[X.index, "Group"] = model.fit_predict(X_scaled).astype(str)
            st.success(f"Records grouped into **{n_clusters}** groups.")
            group_summary = df[["Group"] + numeric_cols_list].groupby("Group").mean().round(2)
            st.dataframe(group_summary, use_container_width=True)
            with st.spinner("AI is analyzing the results..."):
                ai_text = ask_ai(
                    system_prompt="You are a business analyst. Explain ML results in 2-3 short sentences max. Be direct and actionable. No bullet points, no headers.",
                    user_prompt=f"I clustered the data into {n_clusters} groups. Here are the group averages:\n{group_summary.to_string()}\nExplain what each group represents and what the user should do next."
                )
            st.info("💡 " + ai_text)

    elif ml_option == "Find Unusual Records":
        st.markdown("This will highlight records that look different from the rest.")
        contamination = st.slider("Sensitivity (higher = more flagged)", 0.01, 0.2, 0.05, key="contam")
        if st.button("Run", key="run_anomaly") and numeric_cols_list:
            X = df[numeric_cols_list].dropna()
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X)
            anomalies = df.loc[X.index][preds == -1]
            st.warning(f"Found **{len(anomalies)}** unusual records out of {len(X)}.")
            st.dataframe(anomalies, use_container_width=True)
            with st.spinner("AI is analyzing the results..."):
                ai_text = ask_ai(
                    system_prompt="You are a business analyst. Explain ML results in 2-3 short sentences max. Be direct and actionable. No bullet points, no headers.",
                    user_prompt=f"Anomaly detection found {len(anomalies)} unusual records out of {len(X)} total. Sample of anomalies:\n{anomalies.head(5).to_string()}\nExplain what this means and what the user should do next."
                )
            st.info("💡 " + ai_text)

    # =========================
    # 📊 GENERATE DASHBOARD BTN
    # =========================
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("📊 Generate a Dashboard", use_container_width=True, key="go_dashboard"):
            st.session_state.df_for_dashboard = df
            st.session_state.ml_done = True
            st.session_state.page = "dashboard"
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
            default_col = kpi.get("COLUMN", all_cols_db[0])
            default_metric = kpi.get("METRIC", "sum")
            default_label = kpi.get("LABEL", f"KPI {i+1}")
            label_inp = st.text_input("Label", value=default_label, key=f"kpi_label_{i}")
            col_sel = st.selectbox("Column",
                                   options=all_cols_db,
                                   index=all_cols_db.index(default_col) if default_col in all_cols_db else 0,
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
            color_col = resolve_col(color_col) if color_col and color_col != "none" else None
            color_col = color_col if color_col in df.columns else None

            # fallback if y_col not found → use first numeric col
            if y_col not in df.columns and y_col != "count":
                num_fallback = [c for c in df.select_dtypes(include="number").columns if c != x_col]
                y_col = num_fallback[0] if num_fallback else "count"
            st.markdown(f"<div style='color:#a0a8ff;font-size:0.85rem;font-weight:600;margin-bottom:4px'>{title}</div>", unsafe_allow_html=True)
            try:
                plot_df = df.copy()
                # Remove Unknown values
                for c in plot_df.select_dtypes(include="object").columns:
                    plot_df = plot_df[plot_df[c] != "Unknown"]
                if x_col in plot_df.columns:
                    try:
                        # Only convert to date if column name suggests it's a date
                        col_lower_x = x_col.lower()
                        date_hints = ["date", "time", "day", "month", "year", "created", "updated", "birth", "timestamp"]
                        if any(k in col_lower_x for k in date_hints):
                            parsed = pd.to_datetime(plot_df[x_col], errors="coerce")
                            if parsed.notna().mean() > 0.7:
                                plot_df[x_col] = parsed.dt.to_period("M").astype(str)
                    except: pass
                if ctype == "pie":
                    grp = plot_df[x_col].value_counts().head(7).reset_index()
                    grp.columns = [x_col, "count"]
                    fig = px.pie(grp, names=x_col, values="count",
                                 color_discrete_sequence=px.colors.sequential.Purples_r, hole=0.4)
                elif ctype == "scatter":
                    fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col,
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                elif ctype == "histogram":
                    fig = px.histogram(plot_df, x=x_col, color=color_col,
                                       color_discrete_sequence=["#7c83fd"])
                elif ctype == "line":
                    if y_col == "count":
                        grp = plot_df[x_col].value_counts().sort_index().reset_index()
                        grp.columns = [x_col, "count"]
                        fig = px.line(grp, x=x_col, y="count", color_discrete_sequence=["#7c83fd"])
                    else:
                        grp = plot_df.groupby(x_col)[y_col].sum().reset_index()
                        fig = px.line(grp, x=x_col, y=y_col, color_discrete_sequence=["#7c83fd"])
                else:
                    if y_col == "count":
                        grp = plot_df[x_col].value_counts().head(10).reset_index()
                        grp.columns = [x_col, "count"]
                        fig = px.bar(grp, x=x_col, y="count", color="count", color_continuous_scale="Blues")
                    else:
                        grp = plot_df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(10)
                        fig = px.bar(grp, x=x_col, y=y_col, color=color_col,
                                     color_continuous_scale="Blues" if not color_col else None)
                fig.update_layout(
                    plot_bgcolor="#13151f", paper_bgcolor="#13151f",
                    font_color="#e0e0e0", coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10), height=height,
                    showlegend=True, legend=dict(font=dict(size=10))
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"Could not render: {e}")

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
                <div style='background:linear-gradient(135deg,#111627 0%,#151c35 100%);border:1px solid #2d3555;border-radius:16px;padding:24px 20px;position:relative;overflow:hidden'>
                    <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#7c83fd,#a78bfa)'></div>
                    <div style='color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;font-weight:600'>{label}</div>
                    <div style='color:#f1f5f9;font-size:2rem;font-weight:800;line-height:1;letter-spacing:-1px'>{val_fmt}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # AI generates chart plan
        if "dashboard_charts" not in st.session_state or st.session_state.dashboard_charts is None:
            with st.spinner("AI is building your charts..."):
                st.session_state.dashboard_charts = ask_ai(
                    system_prompt=f"""You are a BI dashboard designer. Suggest exactly 4 professional charts.
Return in this strict format:
TITLE: <chart title>
TYPE: <bar|line|pie|scatter|histogram>
X: <column name>
Y: <numeric column name or 'count'>
COLOR: <categorical column name or 'none'>
---
Repeat 4 times. Only use columns that exist: {columns}. No extra text.""",
                    user_prompt=f"Summary:\n{summary}\nSample:\n{sample_data}"
                )

        chart_blocks = [b.strip() for b in st.session_state.dashboard_charts.split("---") if b.strip()][:4]

        # ── ROW 1: big chart left (2/3) + tall chart right (1/3) ──
        if len(chart_blocks) >= 2:
            r1_left, r1_right = st.columns([2, 1])
            with r1_left:
                with st.container():
                    st.markdown("<div style='background:linear-gradient(135deg,#111627,#131929);border:1px solid #2d3555;border-radius:16px;padding:20px'>", unsafe_allow_html=True)
                    render_chart(chart_blocks[0], height=360)
                    st.markdown("</div>", unsafe_allow_html=True)
            with r1_right:
                with st.container():
                    st.markdown("<div style='background:linear-gradient(135deg,#111627,#131929);border:1px solid #2d3555;border-radius:16px;padding:20px'>", unsafe_allow_html=True)
                    render_chart(chart_blocks[1], height=360)
                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 2: two equal charts ──
        if len(chart_blocks) >= 4:
            r2_left, r2_right = st.columns(2)
            with r2_left:
                st.markdown("<div style='background:linear-gradient(135deg,#111627,#131929);border:1px solid #2d3555;border-radius:16px;padding:20px'>", unsafe_allow_html=True)
                render_chart(chart_blocks[2], height=300)
                st.markdown("</div>", unsafe_allow_html=True)
            with r2_right:
                st.markdown("<div style='background:linear-gradient(135deg,#111627,#131929);border:1px solid #2d3555;border-radius:16px;padding:20px'>", unsafe_allow_html=True)
                render_chart(chart_blocks[3], height=300)
                st.markdown("</div>", unsafe_allow_html=True)
