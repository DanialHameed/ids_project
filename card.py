import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib as mpl

# --- Page Config ---
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    html, body, .stApp {
        height: 100%;
        min-height: 100vh;
        background: linear-gradient(135deg, #0a1626 0%, #1a2636 60%, #2a1c41 100%);
        background-attachment: fixed;
        font-family: 'Segoe UI', 'Roboto', 'Montserrat', 'Arial', sans-serif;
    }
    .main, .stApp {
        background: transparent !important;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0C1C2C !important;
        color: #00E5FF !important;
        box-shadow: 0 0 24px #00E5FF33, 0 0 8px #E3F2FD33;
    }
    section[data-testid="stSidebar"] * {
        color: #00E5FF !important;
        text-shadow: 0 0 8px #00E5FF, 0 0 4px #E3F2FD;
    }
    /* Sidebar headings and radio */
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1v0mbdj h1, .st-emotion-cache-1v0mbdj h2, .st-emotion-cache-1v0mbdj h3 {
        color: #00E5FF !important;
        text-shadow: 0 0 12px #00E5FF, 0 0 6px #E3F2FD;
    }
    /* Stat/metric boxes */
    .metric-box, .stMetric {
        background: rgba(0,229,255,0.12) !important;
        color: #00E5FF !important;
        border-radius: 22px;
        box-shadow: 0 2px 16px #00E5FF99, 0 0 8px #E3F2FD99;
        border: 1.5px solid #00E5FF;
        text-shadow: 0 0 8px #00E5FF, 0 0 4px #E3F2FD;
        font-weight: 700;
    }
    /* All headers */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #00E5FF !important;
        text-shadow: 0 0 16px #00E5FF, 0 0 8px #E3F2FD, 0 0 4px #ff00cc;
        font-family: 'Montserrat', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    /* All informative/interactive text blocks */
    p, span, .stMarkdown, .stText, .stDataFrame, .stTabs [data-baseweb="tab"] {
        color: #E3F2FD !important;
        text-shadow: 0 0 8px #00E5FF, 0 0 4px #E3F2FD;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00e5ff 0%, #ff00cc 100%);
        color: #E3F2FD;
        font-weight: 700;
        border-radius: 18px;
        height: 3em;
        width: 100%;
        box-shadow: 0 4px 24px #00E5FF88, 0 1.5px 8px #ff00cc88;
        border: none;
        text-shadow: 0 0 8px #00E5FF, 0 0 4px #E3F2FD;
        letter-spacing: 1px;
        font-size: 1.1em;
        transition: transform 0.1s, box-shadow 0.1s;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.04);
        box-shadow: 0 8px 32px #00E5FFcc, 0 3px 16px #ff00cccc;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 34, 54, 0.88) !important;
        color: #00E5FF !important;
        border-radius: 18px 18px 0 0;
        margin-right: 4px;
        font-weight: 600;
        font-size: 1.1em;
        border: 1.5px solid #ff00cc;
        box-shadow: 0 2px 8px #00E5FF55;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00E5FF 0%, #ff00cc 100%) !important;
        color: #232526 !important;
        box-shadow: 0 2px 16px #00E5FF99, 0 2px 8px #ff00cc99;
        border-bottom: 2.5px solid #ff00cc;
    }
    .stDataFrame {
        background: rgba(30, 34, 54, 0.88) !important;
        color: #E3F2FD !important;
        border-radius: 18px;
        box-shadow: 0 4px 20px #00E5FF55, 0 0 8px #ff00cc55;
        border: 1.5px solid #00E5FF;
    }
    @media (max-width: 900px) {
        .metric-box, .stDataFrame, .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
        }
        .stButton>button {
            border-radius: 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Matplotlib/Seaborn color settings for charts ---
mpl.rcParams['axes.facecolor'] = '#1e2236'
mpl.rcParams['figure.facecolor'] = '#1e2236'
mpl.rcParams['axes.edgecolor'] = '#00f0ff'
mpl.rcParams['axes.labelcolor'] = '#00f0ff'
mpl.rcParams['xtick.color'] = '#bfe9ff'
mpl.rcParams['ytick.color'] = '#bfe9ff'
mpl.rcParams['text.color'] = '#fff'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.titlecolor'] = '#ff00cc'

sns.set_palette(sns.color_palette(["#00f0ff", "#ff00cc", "#3a1c71", "#bfe9ff"]))

# For heatmaps
CYAN_TO_NAVY = sns.color_palette(["#00f0ff", "#232526", "#3a1c71"])

# --- Data & Model Loading ---
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

def get_sampled_df(df, n=5000):
    if len(df) > n:
        return df.sample(n=n, random_state=42)
    return df

@st.cache_resource
def load_model():
    return joblib.load("credit_fraud.pkl")  # Must contain: model, scaler, feature_names

df = load_data()
sampled_df = get_sampled_df(df)
model_bundle = load_model()
model = model_bundle['model']
scaler = model_bundle['scaler']
feature_names = model_bundle['feature_names']  # Don't alter this

# --- Caching for EDA ---
@st.cache_data(show_spinner=False)
def get_corr(df):
    return df.corr()

@st.cache_data(show_spinner=False)
def get_hist_data(df, feature):
    return df[feature], df['Class']

@st.cache_data(show_spinner=False)
def get_box_data(df, feature):
    return df['Class'], df[feature]

@st.cache_data(show_spinner=False)
def get_scatter_data(df, feature):
    return df[feature], df['Amount'], df['Class']

# --- Sidebar Navigation ---
st.sidebar.image("https://img.icons8.com/fluency/96/credit-card.png", width=60)
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "EDA", "Model & Prediction", "Conclusion"])

# --- Introduction Section ---
if section == "Introduction":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("""
    ## About
    This project analyzes credit card transactions to detect fraudulent activity using machine learning.
    The dataset contains transactions labeled as legitimate (0) or fraudulent (1), with features V1-V28 being PCA transformed.

    ### Key Features:
    - Real-time fraud detection
    - Interactive visualizations
    - Model performance metrics
    - User-friendly interface
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraudulent Transactions", f"{df['Class'].sum():,}")
    with col3:
        st.metric("Fraud Percentage", f"{(df['Class'].mean()*100):.2f}%")

    st.subheader("Preview Data")
    st.dataframe(df.head(10))

# --- EDA Section ---
elif section == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("Feature Visualizations")
    feature = st.selectbox("Select Feature for Visualization", [col for col in sampled_df.columns if col != 'Class'])
    # Show all three plots at once, no headings
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        x_hist, hue_hist = get_hist_data(sampled_df, feature)
        sns.histplot(x=x_hist, hue=hue_hist, bins=50, palette='cool', ax=ax1)
        ax1.set_facecolor('#232526')
        fig1.patch.set_facecolor('#232526')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        x_box, y_box = get_box_data(sampled_df, feature)
        sns.boxplot(x=x_box, y=y_box, palette='cool', ax=ax2)
        ax2.set_facecolor('#232526')
        fig2.patch.set_facecolor('#232526')
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        if feature != 'Class':
            x_scatter, y_scatter, hue_scatter = get_scatter_data(sampled_df, feature)
            sns.scatterplot(x=x_scatter, y=y_scatter, hue=hue_scatter, palette='cool', ax=ax3)
        ax3.set_facecolor('#232526')
        fig3.patch.set_facecolor('#232526')
        st.pyplot(fig3)

    st.subheader("Correlation Analysis")
    if st.button("Show Correlation Heatmap and Bar Plot"):
        corr = get_corr(sampled_df)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap=CYAN_TO_NAVY, linewidths=0.5, ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        # Correlation bar plot for a selected feature
        selected_corr_feature = st.selectbox("Select Feature for Correlation Bar Plot", [col for col in sampled_df.columns if col != 'Class'], key='corr_bar')
        corr_vals = corr[selected_corr_feature].drop(selected_corr_feature).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=corr_vals.values, y=corr_vals.index, palette='cool', ax=ax2)
        plt.title(f'Correlation of All Features with {selected_corr_feature}')
        st.pyplot(fig2)

# --- Model & Prediction Section ---
elif section == "Model & Prediction":
    st.title("🤖 Model & Prediction")

    st.subheader("Model Performance")

    # Safety check
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns in dataset required by model: {missing_cols}")
    else:
        try:
            X = df[feature_names]
            y_true = df['Class']
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
            with col2:
                st.metric("F1 Score", f"{f1:.4f}")
        except Exception as e:
            st.error(f"⚠️ Error during model evaluation: {e}")

    st.subheader("Try a Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        input_data = {}
        for i, feature in enumerate(feature_names):
            with (col1 if i % 2 == 0 else col2):
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=float(df[feature].mean()) if feature in df.columns else 0.0,
                    format="%.6f"
                )
        submitted = st.form_submit_button("🚨 Predict Fraud")
        if submitted:
            try:
                input_df = pd.DataFrame([input_data])[feature_names]
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                # Robust probability handling
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_scaled)[0][1]
                elif hasattr(model, 'decision_function'):
                    # Scale decision_function output to [0,1]
                    decision = model.decision_function(input_scaled)[0]
                    # Min-max scaling for binary classification
                    min_dec, max_dec = model.decision_function(scaler.transform(df[feature_names])).min(), model.decision_function(scaler.transform(df[feature_names])).max()
                    probability = (decision - min_dec) / (max_dec - min_dec) if max_dec > min_dec else 0.5
                else:
                    probability = float('nan')
                st.markdown("### Prediction Result")
                result = "❌ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
                st.markdown(f"**{result}**")
                st.markdown(f"Probability of fraud: **{probability:.2%}**")
            except Exception as e:
                st.error(f"Prediction error: {e}. Please ensure all features are numeric and match the model's expected input.")

# --- Conclusion Section ---
elif section == "Conclusion":
    st.title("📝 Conclusion")
    st.markdown("""
    ### Key Takeaways:
    1. **Data Imbalance**: Fraudulent transactions are rare.
    2. **Model Accuracy**: The model performs well with high accuracy and F1 score.
    3. **Feature Insight**: PCA features help in detecting fraud efficiently.
    4. **App Utility**: This app enables interactive fraud detection testing.

    ### Future Improvements:
    - Add real-time monitoring with API integration
    - Store predictions with timestamps
    - Enable authentication for secure access
    - Export reports and logs
    """)

# --- Footer ---
st.markdown("""<hr style='border: 1px solid gray;'>\n<p style='text-align:center;color:gray;'>Built with ❤️ | Powered by Streamlit</p>""", unsafe_allow_html=True)


