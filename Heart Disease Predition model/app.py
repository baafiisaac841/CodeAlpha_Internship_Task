import streamlit as st
import numpy as np
import pickle
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI · Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8e8f0;
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Hero header ── */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
  }
  .hero h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .3rem;
  }
  .hero p {
    color: #a8a8c8;
    font-size: 1.05rem;
    margin-top: 0;
  }

  /* ── Glass cards ── */
  .glass {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.4rem;
  }
  .section-title {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1rem;
  }

  /* ── Slider & widget labels ── */
  label { color: #d1d1e8 !important; font-size: .88rem !important; }
  .stSlider > div[data-baseweb] > div { background: rgba(167,139,250,.25) !important; }
  .stSlider [data-baseweb="thumb"] { background: #a78bfa !important; }

  /* ── Select boxes ── */
  .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
  }

  /* ── Number inputs ── */
  .stNumberInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
  }

  /* ── Predict button ── */
  .stButton > button {
    width: 100%;
    padding: 1rem 0;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: .05em;
    border: none;
    border-radius: 14px;
    background: linear-gradient(90deg, #f093fb, #f5576c);
    color: #fff;
    cursor: pointer;
    transition: transform .15s, box-shadow .15s;
    box-shadow: 0 8px 30px rgba(245,87,108,.35);
    margin-top: .5rem;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(245,87,108,.5);
  }

  /* ── Result boxes ── */
  .result-danger {
    background: linear-gradient(135deg, rgba(245,87,108,.22), rgba(240,147,251,.15));
    border: 1.5px solid rgba(245,87,108,.55);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: pulse 2s infinite;
  }
  .result-safe {
    background: linear-gradient(135deg, rgba(79,209,197,.18), rgba(72,187,120,.15));
    border: 1.5px solid rgba(79,209,197,.5);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(245,87,108,.35); }
    50%      { box-shadow: 0 0 0 12px rgba(245,87,108,0); }
  }
  .result-title { font-size: 1.7rem; font-weight: 700; margin: .4rem 0; }
  .result-prob  { font-size: 1rem; color: #c8c8e0; margin-top: .5rem; }

  /* ── Metric chips ── */
  .chip-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: 1rem; }
  .chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 30px;
    padding: .35rem 1rem;
    font-size: .82rem;
    color: #c0c0e0;
  }
  .chip b { color: #fff; }

  /* ── Info tip ── */
  .tip {
    background: rgba(167,139,250,.12);
    border-left: 3px solid #a78bfa;
    border-radius: 0 10px 10px 0;
    padding: .7rem 1rem;
    font-size: .83rem;
    color: #b8b8d8;
    margin-top: 1rem;
  }

  /* ── Progress bar colour ── */
  .stProgress > div > div > div > div { background: linear-gradient(90deg,#f093fb,#f5576c) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_heart_disease_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🫀 CardioAI</h1>
  <p>AI-powered heart disease risk assessment · Powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Layout: two columns (form | results) ─────────────────────────────────────
form_col, result_col = st.columns([1.05, 0.95], gap="large")

with form_col:

    # — Demographics —
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">👤 Demographics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age    = st.slider("Age", 18, 85, 45, help="Patient age in years")
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

    # — Clinical measurements —
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🩺 Clinical Measurements</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        glucose     = st.slider("Glucose (mg/dL)", 30, 200, 100)
        systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        bmi         = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)
    with c2:
        cholesterol  = st.slider("Cholesterol (mg/dL)", 100, 350, 180)
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
        heart_rate   = st.slider("Heart Rate (bpm)", 45, 120, 75)
    st.markdown('</div>', unsafe_allow_html=True)

    # — Lifestyle —
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🏃 Lifestyle & History</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        smoking           = st.selectbox("Smoking", ["No", "Yes"])
        alcohol           = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    with c2:
        physical_activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
        family_history    = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍 Analyse My Risk", use_container_width=True)

# ── Results panel ─────────────────────────────────────────────────────────────
with result_col:
    st.markdown('<div class="glass" style="min-height:520px">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Risk Assessment</p>', unsafe_allow_html=True)

    if predict_btn:
        # Encode inputs exactly as the notebook did
        gender_enc   = 1 if gender == "Male"   else 0
        smoking_enc  = 1 if smoking == "Yes"   else 0
        alcohol_enc  = 1 if alcohol == "Yes"   else 0
        activity_enc = {"Low": 0, "Medium": 1, "High": 2}[physical_activity]
        family_enc   = 1 if family_history == "Yes" else 0

        features = np.array([[age, gender_enc, glucose, cholesterol,
                               systolic_bp, diastolic_bp, bmi, heart_rate,
                               smoking_enc, alcohol_enc, activity_enc, family_enc]])

        with st.spinner("Analysing…"):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.008)
                progress.progress(i + 1)
            progress.empty()

            pred      = model.predict(features)[0]
            proba     = model.predict_proba(features)[0]
            risk_pct  = round(proba[1] * 100, 1)
            safe_pct  = round(proba[0] * 100, 1)

        if pred == 1:
            st.markdown(f"""
            <div class="result-danger">
              <div style="font-size:3.5rem">⚠️</div>
              <div class="result-title" style="color:#f5576c">HIGH RISK DETECTED</div>
              <div class="result-prob">Probability of heart disease: <b style="color:#fda085">{risk_pct}%</b></div>
              <div class="result-prob">Low risk probability: {safe_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
            advice = [
                "🏥 Consult a cardiologist promptly",
                "💊 Review current medications with your doctor",
                "🥗 Adopt a heart-healthy, low-sodium diet",
                "🚭 Quit smoking if applicable",
                "🏃 Begin supervised exercise programme",
            ]
        else:
            st.markdown(f"""
            <div class="result-safe">
              <div style="font-size:3.5rem">✅</div>
              <div class="result-title" style="color:#4fd1c5">LOW RISK</div>
              <div class="result-prob">Probability of heart disease: <b style="color:#68d391">{risk_pct}%</b></div>
              <div class="result-prob">Safe probability: {safe_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
            advice = [
                "✔️ Maintain your current healthy habits",
                "🥦 Keep eating a balanced, nutritious diet",
                "🏋️ Continue regular physical activity",
                "🩺 Schedule annual health check-ups",
                "😴 Prioritise quality sleep (7–9 hrs)",
            ]

        # Risk gauge bar
        st.markdown("#### Risk Level")
        colour = "#f5576c" if risk_pct > 50 else "#fda085" if risk_pct > 25 else "#4fd1c5"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08);border-radius:30px;height:14px;margin-bottom:6px;">
          <div style="width:{risk_pct}%;background:{colour};border-radius:30px;height:14px;
                      transition:width .6s ease;box-shadow:0 0 12px {colour}88;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#888;">
          <span>0% (Safe)</span><span>50%</span><span>100% (Critical)</span>
        </div>
        """, unsafe_allow_html=True)

        # Key factors
        st.markdown("#### Key Input Summary")
        st.markdown(f"""
        <div class="chip-row">
          <span class="chip">Age <b>{age}</b></span>
          <span class="chip">BMI <b>{bmi}</b></span>
          <span class="chip">BP <b>{systolic_bp}/{diastolic_bp}</b></span>
          <span class="chip">Glucose <b>{glucose}</b></span>
          <span class="chip">Cholesterol <b>{cholesterol}</b></span>
          <span class="chip">HR <b>{heart_rate} bpm</b></span>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("#### Recommendations")
        for a in advice:
            st.markdown(f"- {a}")

        st.markdown("""
        <div class="tip">
          ⚠️ <b>Disclaimer:</b> This tool is for informational purposes only and does
          not replace professional medical advice. Always consult a qualified healthcare
          provider for diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; color:#6060a0;">
          <div style="font-size:5rem; margin-bottom:1rem; opacity:.5">🫀</div>
          <p style="font-size:1.05rem;">Fill in the patient details on the left<br>and click <b style="color:#a78bfa">Analyse My Risk</b>.</p>
          <p style="font-size:.83rem; margin-top:1.5rem; opacity:.7;">
            Model: <b>XGBoost</b> · Dataset: <b>1,000 patients</b> · Features: <b>12</b>
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(255,255,255,0.08); margin-top:2rem;">
<p style="text-align:center; color:#404060; font-size:.78rem;">
  CardioAI · Built with Streamlit & XGBoost · For research & educational use only
</p>
""", unsafe_allow_html=True)
