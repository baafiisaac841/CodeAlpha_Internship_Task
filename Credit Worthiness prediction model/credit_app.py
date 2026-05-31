import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Worthiness Predictor",
    page_icon="💳",
    layout="centered"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0f1a2e 50%, #0a0f1e 100%); }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem; font-weight: 800;
        color: #ffffff; letter-spacing: -0.02em;
        line-height: 1.1; margin-bottom: 0.3rem;
    }
    .hero-sub { font-size: 1rem; color: #6b7fa3; margin-bottom: 2rem; font-weight: 300; }
    .accent { color: #4f9eff; }

    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.5rem; margin-bottom: 1.2rem;
    }
    .section-label {
        font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: #4f9eff; margin-bottom: 1rem;
    }

    .result-worthy {
        background: linear-gradient(135deg, #0d2d1a, #0a3d20);
        border: 1px solid #1a6b35; border-radius: 20px;
        padding: 2rem; text-align: center; margin-top: 1.5rem;
    }
    .result-notworthy {
        background: linear-gradient(135deg, #2d0d0d, #3d0a0a);
        border: 1px solid #6b1a1a; border-radius: 20px;
        padding: 2rem; text-align: center; margin-top: 1.5rem;
    }
    .result-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .result-worthy .result-title { color: #4cff91; }
    .result-notworthy .result-title { color: #ff4c4c; }
    .result-subtitle { font-size: 0.95rem; color: #8a9bb5; margin-bottom: 1.2rem; }

    .prob-bar-container {
        background: rgba(255,255,255,0.08); border-radius: 50px;
        height: 10px; width: 100%; margin: 0.5rem 0 1rem 0; overflow: hidden;
    }
    .prob-bar-worthy { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #1a6b35, #4cff91); }
    .prob-bar-notworthy { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #6b1a1a, #ff4c4c); }

    .metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
    .metric-box { flex: 1; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 0.8rem; text-align: center; }
    .metric-value { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #ffffff; }
    .metric-label { font-size: 0.72rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 0.08em; }

    .stButton > button {
        background: linear-gradient(135deg, #4f9eff, #2563eb) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 0.8rem 2rem !important; font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important; font-size: 1rem !important; width: 100% !important;
        box-shadow: 0 4px 20px rgba(79, 158, 255, 0.3) !important;
    }

    .flag-badge {
        display: inline-block; background: rgba(255,76,76,0.15);
        border: 1px solid rgba(255,76,76,0.3); color: #ff4c4c;
        border-radius: 50px; padding: 0.2rem 0.7rem; font-size: 0.75rem; margin: 0.2rem;
    }
    .flag-badge-ok {
        display: inline-block; background: rgba(76,255,145,0.1);
        border: 1px solid rgba(76,255,145,0.2); color: #4cff91;
        border-radius: 50px; padding: 0.2rem 0.7rem; font-size: 0.75rem; margin: 0.2rem;
    }
    .divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'best_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Credit <span class="accent">Worthiness</span><br>Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter applicant details to assess credit eligibility</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ No model found. Place `best_model.pkl` in the same folder as this app.")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────

# Section 1 — Personal Info
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">👤 Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    person_age       = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income    = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    person_gender    = st.selectbox("Gender", ["male", "female"])
with col2:
    person_emp_exp        = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=5)
    person_education      = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
st.markdown('</div>', unsafe_allow_html=True)

# Section 2 — Loan Info
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">💰 Loan Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    loan_amnt            = st.number_input("Loan Amount ($)", min_value=500, value=10000, step=500)
    loan_intent          = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_int_rate        = st.number_input("Interest Rate (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.1)
with col4:
    loan_percent_income  = st.number_input("Loan % of Income (0-1)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    loan_grade           = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
st.markdown('</div>', unsafe_allow_html=True)

# Section 3 — Credit History
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📊 Credit History</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    previous_loan_defaults = st.selectbox("Previous Loan Default", ["No", "Yes"])
with col6:
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
    credit_score               = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
st.markdown('</div>', unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
predict_btn = st.button("⚡ Assess Credit Worthiness")

# ── PREDICTION LOGIC ──────────────────────────────────────────────────────────
if predict_btn and model is not None:

    # ── STEP 1: Encode exactly as done during training ────────────────────────

    # LabelEncoder for gender (male=1, female=0 — typical sklearn LabelEncoder order)
    gender_map = {'female': 0, 'male': 1}

    # OrdinalEncoder for education
    education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}

    # LabelEncoder for previous defaults
    default_map = {'No': 0, 'Yes': 1}

    # ── STEP 2: Build base numeric features ──────────────────────────────────
    input_dict = {
        'person_age':                   person_age,
        'person_gender':                gender_map[person_gender],
        'person_education':             education_map[person_education],
        'person_income':                person_income,
        'person_emp_exp':               person_emp_exp,
        'loan_amnt':                    loan_amnt,
        'loan_int_rate':                loan_int_rate,
        'loan_percent_income':          loan_percent_income,
        'cb_person_cred_hist_length':   cb_person_cred_hist_length,
        'credit_score':                 credit_score,
        'previous_loan_defaults_on_file': default_map[previous_loan_defaults],
    }

    # ── STEP 3: One-Hot Encode person_home_ownership (get_dummies) ────────────
    for val in ['MORTGAGE', 'OTHER', 'OWN', 'RENT']:
        input_dict[f'person_home_ownership_{val}'] = 1 if person_home_ownership == val else 0

    # ── STEP 4: One-Hot Encode loan_intent (get_dummies) ─────────────────────
    for val in ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']:
        input_dict[f'loan_intent_{val}'] = 1 if loan_intent == val else 0

    # ── STEP 5: Create DataFrame ──────────────────────────────────────────────
    input_data = pd.DataFrame([input_dict])

    # ── STEP 6: Add outlier flags ─────────────────────────────────────────────
    input_data['high_debt_flag']   = int(loan_percent_income > 0.3)
    input_data['zero_income_flag'] = int(person_income == 0)

    # ── STEP 7: Predict ───────────────────────────────────────────────────────
    try:
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        prob_worthy    = round(probability[1] * 100, 1)
        prob_notworthy = round(probability[0] * 100, 1)

        # ── RESULT DISPLAY ────────────────────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="result-worthy">
                <div class="result-title">✅ Credit Worthy</div>
                <div class="result-subtitle">This applicant is recommended for credit approval</div>
                <div style="font-size:0.8rem; color:#6b7fa3; margin-bottom:0.3rem;">Approval Confidence</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-worthy" style="width:{prob_worthy}%"></div>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-value" style="color:#4cff91">{prob_worthy}%</div>
                        <div class="metric-label">Worthy Probability</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" style="color:#ff4c4c">{prob_notworthy}%</div>
                        <div class="metric-label">Risk Probability</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{loan_grade}</div>
                        <div class="metric-label">Loan Grade</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-notworthy">
                <div class="result-title">❌ Not Credit Worthy</div>
                <div class="result-subtitle">This applicant is not recommended for credit approval</div>
                <div style="font-size:0.8rem; color:#6b7fa3; margin-bottom:0.3rem;">Risk Level</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-notworthy" style="width:{prob_notworthy}%"></div>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-value" style="color:#ff4c4c">{prob_notworthy}%</div>
                        <div class="metric-label">Risk Probability</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" style="color:#4cff91">{prob_worthy}%</div>
                        <div class="metric-label">Worthy Probability</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{loan_grade}</div>
                        <div class="metric-label">Loan Grade</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── RISK FLAGS ────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">🚩 Risk Flag Analysis</div>', unsafe_allow_html=True)

        flags_html = ""
        flags_html += '<span class="flag-badge">⚠️ Zero Income</span>'          if person_income == 0          else '<span class="flag-badge-ok">✓ Income Present</span>'
        flags_html += '<span class="flag-badge">⚠️ High Loan-to-Income</span>'  if loan_percent_income > 0.5   else '<span class="flag-badge-ok">✓ Loan-to-Income OK</span>'
        flags_html += '<span class="flag-badge">⚠️ Previous Default</span>'     if previous_loan_defaults=="Yes" else '<span class="flag-badge-ok">✓ No Prior Default</span>'
        flags_html += '<span class="flag-badge">⚠️ High Interest Rate</span>'   if loan_int_rate > 20          else '<span class="flag-badge-ok">✓ Interest Rate Normal</span>'
        flags_html += '<span class="flag-badge">⚠️ No Experience</span>'        if person_emp_exp < 1          else '<span class="flag-badge-ok">✓ Stable Employment</span>'
        flags_html += '<span class="flag-badge">⚠️ Thin Credit History</span>'  if cb_person_cred_hist_length < 2 else '<span class="flag-badge-ok">✓ Credit History OK</span>'
        flags_html += '<span class="flag-badge">⚠️ Low Credit Score</span>'     if credit_score < 580          else '<span class="flag-badge-ok">✓ Credit Score OK</span>'

        st.markdown(f'<div style="margin-top:0.5rem">{flags_html}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

elif predict_btn and model is None:
    st.error("⚠️ Please load your model first — place best_model.pkl in the same folder.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#3a4a6b; font-size:0.78rem;">
    Credit Worthiness Predictor · Built with Streamlit
</div>
""", unsafe_allow_html=True)
