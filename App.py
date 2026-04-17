import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load dataset
@st.cache_data
def load_dataset():
    return pd.read_csv("job_salary_prediction_dataset.csv")

df = load_dataset()

# Load trained models and transformers (with error handling)
@st.cache_resource
def load_models():
    try:
        regressor = joblib.load("best_regressor.pkl")
        classifier = joblib.load("best_classifier.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        return regressor, classifier, scaler, encoders
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Using mock predictions for demo.")
        return None, None, None, None

# Mock prediction functions
def mock_predict_salary(job_title, experience_years, education_level,
                         industry, company_size, location, remote_work):
    base_salary = 80000
    base_salary += experience_years * 3500
    edu_mult = {"High School": 0.85, "Diploma": 0.95, "Bachelor": 1.0, "Master": 1.2, "PhD": 1.45}.get(education_level, 1.0)
    job_mult = {
        "AI Engineer": 1.5, "Machine Learning Engineer": 1.45, "Data Scientist": 1.4,
        "Cloud Engineer": 1.3, "Backend Developer": 1.2, "Frontend Developer": 1.15,
        "DevOps Engineer": 1.25, "Cybersecurity Analyst": 1.2, "Software Engineer": 1.2,
        "Data Analyst": 1.1, "Business Analyst": 1.05, "Product Manager": 1.3
    }.get(job_title, 1.0)
    size_mult = {"Startup": 0.9, "Small": 0.95, "Medium": 1.0, "Large": 1.1, "Enterprise": 1.2}.get(company_size, 1.0)
    remote_mult = {"Yes": 1.05, "Hybrid": 1.02, "No": 1.0}.get(remote_work, 1.0)
    base_salary *= edu_mult * job_mult * size_mult * remote_mult
    base_salary += (np.random.random() - 0.5) * 15000
    return max(30000, int(base_salary))

def mock_predict_level(salary):
    if salary < 119358:
        return "Low", np.random.uniform(0.82, 0.94)
    elif salary < 169492:
        return "Medium", np.random.uniform(0.78, 0.93)
    else:
        return "High", np.random.uniform(0.85, 0.96)

# Page configuration
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS BLOCK ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Reset & Base ── */
    * { box-sizing: border-box; }

    .stApp {
        background: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .main > div { padding: 0; max-width: none; }

    /* ── Header ── */
    .header-container {
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 1.25rem 0;
        position: sticky;
        top: 0;
        z-index: 100;
        margin-bottom: 2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .header-icon {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        padding: 0.85rem;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    }

    .header-text h1 {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        font-size: 1.9rem !important;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }

    .header-text p {
        color: #64748b;
        margin: 0.2rem 0 0 0;
        font-size: 0.825rem;
        font-weight: 400;
    }

    /* ── Main container ── */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem 3rem 2rem;
    }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        padding: 1.75rem;
        height: fit-content;
    }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.35rem;
    }

    .section-icon { color: #3b82f6; font-size: 1.2rem; }

    .section-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.825rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* ── Field labels ── */
    .form-field { margin-bottom: 1.4rem; }

    .field-label {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.45rem;
        font-weight: 600;
        color: #1e293b;
        font-size: 0.825rem;
        letter-spacing: 0.01em;
    }

    .field-icon { color: #94a3b8; font-size: 0.95rem; }

    /* ── Sliders ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
        height: 6px !important;
        border-radius: 3px !important;
    }
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid #3b82f6 !important;
        width: 22px !important;
        height: 22px !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(59,130,246,0.35) !important;
        transition: all 0.15s ease !important;
    }
    .stSlider > div > div > div > div > div:hover {
        box-shadow: 0 0 0 10px rgba(59,130,246,0.12) !important;
        transform: scale(1.08) !important;
    }
    .stSlider > div > div > div > div > div:active {
        transform: scale(0.95) !important;
    }

    /* ── Selectboxes ── */
    .stSelectbox > div > div {
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px !important;
        background: #f8fafc !important;
        font-size: 0.85rem !important;
        color: #1e293b !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
        background: #ffffff !important;
    }

    /* ── Prediction display ── */
    .prediction-display {
        background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%);
        border: 1.5px solid #bfdbfe;
        border-radius: 14px;
        padding: 1.75rem;
        text-align: center;
        margin-bottom: 1.25rem;
    }

    .prediction-icon {
        font-size: 2.75rem;
        color: #2563eb;
        margin-bottom: 0.75rem;
        display: block;
    }

    .prediction-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.35rem;
    }

    .prediction-subtitle {
        color: #3b5fc4;
        font-size: 0.8rem;
        line-height: 1.4;
    }

    /* ── Predict button ── */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1.5rem !important;
        width: 100% !important;
        height: 3rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(99,102,241,0.4) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
    }

    /* ── Result display ── */
    .result-display {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1.25rem;
    }

    .result-amount {
        font-size: 2.2rem;
        font-weight: 800;
        color: #16a34a;
        margin-bottom: 0.35rem;
        letter-spacing: -1px;
    }

    .result-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #166534;
    }

    /* ── Level badges ── */
    .level-badge {
        background: #2563eb;
        color: white;
        padding: 0.55rem 1.25rem;
        border-radius: 100px;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    }
    .level-badge.high {
        background: linear-gradient(135deg, #16a34a, #15803d);
        box-shadow: 0 2px 8px rgba(22,163,74,0.3);
    }
    .level-badge.medium {
        background: linear-gradient(135deg, #d97706, #b45309);
        box-shadow: 0 2px 8px rgba(217,119,6,0.3);
    }
    .level-badge.low {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        box-shadow: 0 2px 8px rgba(220,38,38,0.3);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
    }

    /* ── Hide Streamlit chrome ── */
    .stDeployButton { display: none; }
    footer { visibility: hidden; }
    .stApp > header { visibility: hidden; }
    .stMainBlockContainer { padding-top: 0; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .main-container { padding: 0 1rem 2rem 1rem; }
        .header-content { padding: 0 1rem; }
        .card { margin-bottom: 1rem; }
        .header-text h1 { font-size: 1.4rem !important; }
    }
</style>
""", unsafe_allow_html=True)
# ── END CSS BLOCK ────────────────────────────────────────────────────────────

# Initialize session state
if 'prediction_mode' not in st.session_state:
    st.session_state.prediction_mode = 'salary'

# Header
st.markdown("""
<div class="header-container">
    <div class="header-content">
        <div class="header-icon">
            💼
        </div>
        <div class="header-text">
            <h1>AI Salary Predictor</h1>
            <p>Predict employee compensation with machine learning</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
regressor, classifier, scaler, encoders = load_models()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="section-header">
            <span class="section-icon">💼</span>
            <h2 class="section-title">Employee Details</h2>
        </div>
        <p class="section-subtitle">Enter the employee information to get salary predictions</p>
    """, unsafe_allow_html=True)

    # Job Title
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">🧑‍💻</span>Job Title</div>', unsafe_allow_html=True)
    job_title = st.selectbox("", [
        "Select job title",
        "AI Engineer", "Backend Developer", "Business Analyst",
        "Cloud Engineer", "Cybersecurity Analyst", "Data Analyst",
        "Data Scientist", "DevOps Engineer", "Frontend Developer",
        "Machine Learning Engineer", "Product Manager", "Software Engineer"
    ], key="job_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Experience slider
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    exp_val = st.session_state.get("experience_slider", 5)
    st.markdown(f'<div class="field-label"><span class="field-icon">📈</span>Years of Experience: <strong>{exp_val} yrs</strong></div>', unsafe_allow_html=True)
    experience_years = st.slider("", min_value=0, max_value=40, value=5, key="experience_slider", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Education Level
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">🎓</span>Education Level</div>', unsafe_allow_html=True)
    education_level = st.selectbox("", [
        "Select education level", "High School", "Diploma", "Bachelor", "Master", "PhD"
    ], key="education_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Industry
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">🏭</span>Industry</div>', unsafe_allow_html=True)
    industry = st.selectbox("", [
        "Select industry", "Consulting", "Education", "Finance", "Government",
        "Healthcare", "Manufacturing", "Media", "Retail", "Technology", "Telecom"
    ], key="industry_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Company Size
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">🏢</span>Company Size</div>', unsafe_allow_html=True)
    company_size = st.selectbox("", [
        "Select company size", "Startup", "Small", "Medium", "Large", "Enterprise"
    ], key="company_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Location
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">📍</span>Location</div>', unsafe_allow_html=True)
    location = st.selectbox("", [
        "Select location", "Australia", "Canada", "Germany", "India", "Netherlands",
        "Remote", "Singapore", "Sweden", "UK", "USA"
    ], key="location_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Remote Work
    st.markdown('<div class="form-field">', unsafe_allow_html=True)
    st.markdown('<div class="field-label"><span class="field-icon">🌐</span>Remote Work</div>', unsafe_allow_html=True)
    remote_work = st.selectbox("", [
        "Select remote work type", "Yes", "Hybrid", "No"
    ], key="remote_select", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close card

with col2:
    st.markdown("""
    <div class="card">
        <div class="section-header">
            <span class="section-icon">📊</span>
            <h2 class="section-title">Prediction Results</h2>
        </div>
        <p class="section-subtitle">Choose your prediction type and get AI-powered insights</p>
    """, unsafe_allow_html=True)

    # Tab buttons
    col_tab1, col_tab2 = st.columns(2)

    with col_tab1:
        if st.button("💲 Salary Amount", key="salary_tab", use_container_width=True):
            st.session_state.prediction_mode = 'salary'

    with col_tab2:
        if st.button("🏷️ Salary Level", key="level_tab", use_container_width=True):
            st.session_state.prediction_mode = 'level'

    # Prediction display area
    if st.session_state.prediction_mode == 'salary':
        st.markdown("""
        <div class="prediction-display">
            <span class="prediction-icon">💲</span>
            <div class="prediction-title">Salary Prediction</div>
            <div class="prediction-subtitle">Get the exact predicted salary amount</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-display" style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-color: #86efac;">
            <span class="prediction-icon" style="color: #16a34a;">🏷️</span>
            <div class="prediction-title" style="color: #14532d;">Salary Level</div>
            <div class="prediction-subtitle" style="color: #166534;">Classify salary as Low, Medium, or High</div>
        </div>
        """, unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("✨ Predict Salary", key="predict_btn", use_container_width=True, type="primary")

    # Handle prediction results
    if predict_clicked:
        required = [job_title, education_level, industry, company_size, location, remote_work]
        placeholders = ["Select job title", "Select education level", "Select industry",
                        "Select company size", "Select location", "Select remote work type"]
        if any(v in placeholders for v in required):
            st.error("⚠️ Please fill in all required fields!")
        else:
            with st.spinner("🤖 AI is analyzing the data..."):
                import time
                time.sleep(1.2)

                try:
                    if regressor and classifier and scaler and encoders:
                        input_dict = {
                            "job_title": job_title,
                            "experience_years": experience_years,
                            "education_level": education_level,
                            "industry": industry,
                            "company_size": company_size,
                            "location": location,
                            "remote_work": remote_work,
                        }

                        feature_cols = ['job_title','experience_years','education_level',
                                        'industry','company_size','location','remote_work']

                        input_df = pd.DataFrame([input_dict])[feature_cols]
                        input_df_encoded = input_df.copy()

                        cat_cols = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
                        for col in cat_cols:
                            if col in encoders:
                                input_df_encoded[col] = encoders[col].transform(input_df_encoded[col])

                        input_df_encoded = input_df_encoded[feature_cols]
                        input_scaled = scaler.transform(input_df_encoded)

                        if st.session_state.prediction_mode == 'salary':
                            salary_pred = regressor.predict(input_scaled)[0]
                            st.markdown(f"""
                            <div class="result-display">
                                <div class="result-amount">${salary_pred:,.0f}</div>
                                <div class="result-label">Predicted Annual Salary</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            class_pred = classifier.predict(input_scaled)[0]
                            class_proba = classifier.predict_proba(input_scaled)[0]
                            confidence = max(class_proba) * 100

                            label_map = {0: "Low", 1: "Medium", 2: "High"}
                            predicted_level = label_map.get(class_pred, 'Unknown')
                            badge_class = predicted_level.lower()

                            st.markdown(f"""
                            <div class="result-display">
                                <div class="level-badge {badge_class}">{predicted_level} Salary Level</div><br>
                                <div class="result-label">Confidence: {confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                    else:
                        # Mock predictions
                        if st.session_state.prediction_mode == 'salary':
                            salary_pred = mock_predict_salary(job_title, experience_years, education_level,
                                                               industry, company_size,
                                                               location, remote_work)
                            st.markdown(f"""
                            <div class="result-display">
                                <div class="result-amount">${salary_pred:,}</div>
                                <div class="result-label">Predicted Annual Salary</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            mock_salary = mock_predict_salary(job_title, experience_years, education_level,
                                                               industry, company_size,
                                                               location, remote_work)
                            predicted_level, confidence = mock_predict_level(mock_salary)
                            badge_class = predicted_level.lower()

                            st.markdown(f"""
                            <div class="result-display">
                                <div class="level-badge {badge_class}">{predicted_level} Salary Level</div><br>
                                <div class="result-label">Confidence: {confidence*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close card

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by advanced machine learning algorithms • Built with Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main container