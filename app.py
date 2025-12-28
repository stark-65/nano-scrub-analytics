import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import io
import time
import sys  # Added for capturing Python Sandbox output

# =================================================================
# 1. APP CONFIGURATION & BRANDING
# =================================================================
APP_NAME = "NANO SCRUB - ADVANCED ANALYTICS"
DEVELOPER = "ANKIT JHA"
VERSION = "v3.5 (ML Edition)"

st.set_page_config(
    page_title=APP_NAME, 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# 2. PROFESSIONAL CSS INJECTION (NEON GLASS THEME)
# =================================================================
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #0e1117; color: #ffffff; }
    
    /* FIX: Glassmorphism for Metric Boxes (Prevents White-on-White issues) */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px);
        padding: 25px !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    
    /* Neon Typography for Data Metrics */
    div[data-testid="stMetricValue"] > div { 
        color: #00fbff !important; 
        font-family: 'Orbitron', sans-serif;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] > div { 
        color: #e0e0e0 !important; 
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { 
        height: 55px; 
        background-color: #161b22; 
        color: #ffffff;
        border-radius: 10px; 
        padding: 12px 24px; 
        font-weight: 600;
        border: 1px solid #30363d;
        transition: all 0.4s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { 
        border-color: #00fbff;
        color: #00fbff;
        transform: translateY(-2px);
    }
    
    /* Button Aesthetics */
    .stButton>button { 
        width: 100%; 
        border-radius: 25px; 
        background: linear-gradient(135deg, #0d47a1 0%, #00fbff 100%); 
        color: white; 
        font-weight: bold; 
        border: none; 
        height: 45px;
        box-shadow: 0 4px 15px rgba(0, 251, 255, 0.2);
    }
    .stButton>button:hover { 
        box-shadow: 0 0 25px rgba(0, 251, 255, 0.5);
        transform: scale(1.02);
    }

    /* NEON CYBERPUNK TERMINAL FOR CODING */
    .stTextArea textarea {
        font-family: 'Fira Code', 'Cascadia Code', monospace !important;
        background-color: #000000 !important;
        color: #00fbff !important; 
        border: 1px solid #00fbff !important;
        text-shadow: 0 0 10px rgba(0, 251, 255, 0.4);
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 3. SIDEBAR NAVIGATION & TOOLS
# =================================================================
with st.sidebar:
    st.title(f"🧬 {APP_NAME}")
    st.markdown(f"**Developer:** `{DEVELOPER}`")
    st.markdown(f"**Version:** `{VERSION}`")
    st.markdown("---")
    
    st.header("🔑 Authentication")
    google_key = st.text_input("Gemini AI Gateway Key", type="password", help="Enter your Google AI API Key here.")
    
    st.header("⚙️ Transformation Logic")
    remove_outliers = st.checkbox("Apply IQR Outlier Removal", value=False)
    fill_missing = st.selectbox(
        "Null Handling Strategy", 
        ["None", "Mean/Mode", "Drop Rows", "Zero Fill"]
    )
    
    st.header("🧪 Advanced Engines")
    enable_sandbox = st.toggle("Activate Python Lab", help="Enables the execution of custom scripts on the data.")
    
    st.markdown("---")
    st.info("💡 **Developer Note:** The Sandbox now supports `scikit-learn`. You can build models like K-Means or Linear Regression live.")
    st.caption(f"© 2025 {DEVELOPER} | Deep Analytics Suite")

# =================================================================
# 4. CORE DATA PROCESSING FUNCTIONS
# =================================================================
def clean_data(df, strategy, outliers):
    """Handles missing values and statistical outliers based on user input."""
    # Handle Missing Values
    if strategy == "Mean/Mode":
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
    elif strategy == "Drop Rows":
        df = df.dropna()
    elif strategy == "Zero Fill":
        df = df.fillna(0)

    # Handle Outliers using Interquartile Range (IQR)
    if outliers:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            # Filter rows that fall outside 1.5 * IQR
            df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

# =================================================================
# 5. MAIN APPLICATION LOGIC
# =================================================================
st.title(f"🚀 {APP_NAME}")
st.markdown(f"**The Ultimate Data Science Ecosystem** | Engineered by {DEVELOPER}")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload Data Asset (CSV or XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    # Processing Animation
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.003)
        progress_bar.progress(i + 1)
    
    try:
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
        
        # Apply Transformation Engine
        df = clean_data(raw_df.copy(), fill_missing, remove_outliers)
        
        # DISPLAY PERFORMANCE METRICS
        st.write("### 💎 Data Integrity Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", len(df), delta=f"{len(df)-len(raw_df)} filtered")
        m2.metric("Total Features", len(df.columns))
        m3.metric("Numeric Features", len(df.select_dtypes(include=np.number).columns))
        m4.metric("Missing Cells", df.isna().sum().sum())

        # TABBED NAVIGATION SYSTEM
        tabs = st.tabs([
            "📊 Data Explorer", 
            "📉 Stats Lab", 
            "🎨 Visual Studio", 
            "🧠 AI Analyst", 
            "💻 Python Lab", 
            "🧹 Export Center"
        ])

        # --- TAB 1: DATA EXPLORER ---
        with tabs[0]: 
            st.subheader("High-Level Data Inspection")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write("**Preview (First 25 Rows)**")
                st.dataframe(df.head(25), use_container_width=True)
            with col_b:
                st.write("**Feature Metadata**")
                info_df = pd.DataFrame({
                    "Type": df.dtypes,
                    "Nulls": df.isnull().sum(),
                    "Unique": df.nunique()
                })
                st.table(info_df)

        # --- TAB 2: STATS LAB ---
        with tabs[1]: 
            st.subheader("Statistical Profiling Engine")
            num_df = df.select_dtypes(include=np.number)
            if not num_df.empty:
                st.write("**Descriptive Statistics**")
                st.write(num_df.describe())
                st.write("---")
                st.write("**Pearson Correlation Heatmap**")
                corr = num_df.corr()
                fig_corr = px.imshow(
                    corr, text_auto=True, 
                    color_continuous_scale='RdBu_r', 
                    title="Feature Relationship Matrix",
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Insufficient numeric data found for statistical profiling.")

        # --- TAB 3: VISUAL STUDIO ---
        with tabs[2]: 
            st.subheader("Dynamic Visualization Studio")
            v_col1, v_col2 = st.columns([1, 3])
            with v_col1:
                v_type = st.radio("Chart Category", ["Scatter", "Bar", "Line", "Histogram", "Boxplot"])
                x_ax = st.selectbox("Assign X Axis", df.columns)
                y_ax = st.selectbox("Assign Y Axis", df.columns)
                color_ax = st.selectbox("Color Mapping (Optional)", [None] + list(df.columns))
            
            with v_col2:
                if v_type == "Scatter":
                    fig = px.scatter(df, x=x_ax, y=y_ax, color=color_ax, template="plotly_dark")
                elif v_type == "Bar":
                    fig = px.bar(df, x=x_ax, y=y_ax, color=color_ax, barmode="group")
                elif v_type == "Line":
                    fig = px.line(df, x=x_ax, y=y_ax, color=color_ax)
                elif v_type == "Histogram":
                    fig = px.histogram(df, x=x_ax, color=color_ax, marginal="box")
                elif v_type == "Boxplot":
                    fig = px.box(df, x=x_ax, y=y_ax, color=color_ax)
                st.plotly_chart(fig, use_container_width=True)

        # --- TAB 4: AI ANALYST ---
        with tabs[3]: 
            st.subheader("GenAI Deep-Dive Engine")
            if not google_key:
                st.warning("⚠️ Access Denied: Please provide your Gemini API Key in the sidebar.")
            else:
                query = st.text_area("Specify Research Parameters:", placeholder="Example: Analyze the correlation between feature X and Y...")
                if st.button("Generate Expert Intelligence Report"):
                    try:
                        genai.configure(api_key=google_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt_context = f"""
                        App: {APP_NAME}. 
                        Dataset Schema: {list(df.columns)}. 
                        Statistics Summary: {num_df.describe().to_string()}. 
                        User Inquiry: {query}
                        """
                        with st.spinner("Analyzing data patterns via Gemini Neural Network..."):
                            response = model.generate_content(prompt_context)
                            st.markdown("### 🧬 AI Intelligence Insight")
                            st.success(response.text)
                    except Exception as e:
                        st.error(f"AI Neural Link Error: {e}")

        # --- TAB 5: PYTHON LAB (SANDBOX) ---
        with tabs[4]: 
            st.subheader("💻 Integrated Python Sandbox")
            if not enable_sandbox:
                st.warning("Critical: The Sandbox is currently offline. Toggle 'Enable Python Lab' in the sidebar.")
            else:
                st.info("Direct access enabled. Use `df` to interact with your data. Libraries: `pd`, `np`, `plt`, `px`, `sklearn`.")
                
                # Preset Logic Buttons for User
                st.write("**Quick Logic Presets**")
                c1, c2, c3 = st.columns(3)
                if c1.button("Build K-Means Model"):
                    code_val = "# Build a 3-Cluster Model\nfrom sklearn.cluster import KMeans\nX = df.select_dtypes(include=[np.number])\nkmeans = KMeans(n_clusters=3).fit(X)\ndf['Clusters'] = kmeans.labels_\nprint('Clustering Complete. See Data Preview.')"
                elif c2.button("Run Data Summary"):
                    code_val = "print('--- SCHEMA INFO ---')\nprint(df.info())\nprint('\\n--- DESCRIPTIVE STATS ---')\nprint(df.describe())"
                elif c3.button("Reset Script"):
                    code_val = "print(df.head())"
                else:
                    code_val = "print(df.head())"

                code_input = st.text_area("Script Terminal", height=300, value=code_val)
                
                if st.button("🚀 Execute Script"):
                    buffer = io.StringIO()
                    sys.stdout = buffer
                    try:
                        # PASSING FULL DATA SCIENCE STACK TO THE SANDBOX
                        import sklearn
                        exec_env = {
                            'df': df, 'pd': pd, 'np': np, 'plt': plt, 
                            'px': px, 'sns': sns, 'st': st, 'sklearn': sklearn
                        }
                        exec(code_input, {}, exec_env)
                        sys.stdout = sys.__stdout__ 
                        
                        # Display Results
                        output = buffer.getvalue()
                        st.markdown("### 📥 Terminal Output")
                        st.code(output if output else "Execution Successful (No Output).", language='text')
                        
                        if 'df' in exec_env:
                            df = exec_env['df']
                            st.markdown("### 📑 Modified Data Preview")
                            st.dataframe(df.head())
                    except Exception as e:
                        sys.stdout = sys.__stdout__
                        st.error(f"Logic Error: {e}")

        # --- TAB 6: RAW EXPORT ---
        with tabs[5]: 
            st.subheader("Data Export & Serialization")
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Data Asset (CSV)", 
                data=csv_data, 
                file_name="nanoscrub_export_v35.csv", 
                mime="text/csv"
            )
            st.write("**Final Dataset State:**")
            st.dataframe(df)

    except Exception as e:
        st.error(f"System Critical Failure: {e}")
else:
    # Landing Screen
    st.info(f"🧬 Welcome, {DEVELOPER}. The {APP_NAME} engine is ready. Please upload a dataset to initialize.")

# =================================================================
# 6. FOOTER
# =================================================================
st.markdown("---")
st.caption(f"🛡️ {APP_NAME} | Built by {DEVELOPER} | {VERSION} | Secure Data Environment")