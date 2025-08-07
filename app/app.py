import streamlit as st
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_disease

st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🧠",
    layout="wide"
)

encoders = joblib.load("models/label_encoders.pkl")
symptom_encoder = encoders["Symptom_1"]
all_symptoms = list(symptom_encoder.classes_)

# 💄 Custom CSS — no background, clean layout
st.markdown("""
<style>
/* Remove all background image */
body {
    background: #f6f9fc;
    font-family: 'Segoe UI', sans-serif;
}

/* Main content box */
.main-container {
    background: #ffffff;
    padding: 2rem 3rem;
    margin: 3rem auto;
    width: 80%;
    border-radius: 1rem;
    box-shadow: 0 0 20px rgba(0,0,0,0.05);
}

/* Headings */
h1, h2, h3 {
    color: #0b132b;
    text-align: center;
}

/* Button style */
.stButton > button {
    background: linear-gradient(to right, #2193b0, #6dd5ed);
    color: white;
    font-size: 1.1rem;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    border: none;
}

/* Result box */
.result-box {
    background-color: #e0f7fa;
    padding: 1.2rem;
    border-radius: 10px;
    margin-top: 2rem;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
    border-left: 5px solid #00bcd4;
}
</style>
<div class="main-container">
""", unsafe_allow_html=True)

st.markdown("## 🧠 Welcome to AI Disease Predictor")
st.markdown("#### 🤖 Powered by Machine Learning — Trained on symptoms & diagnoses")
st.markdown("---")

# Symptom selection
cols = st.columns(5)
symptoms = [col.selectbox(f"Symptom {i+1}", all_symptoms, key=f"symptom_{i}") for i, col in enumerate(cols)]

st.markdown("<br>", unsafe_allow_html=True)

centered = st.columns([1, 2, 1])[1]
with centered:
    if st.button("🚀 Predict My Disease"):
        with st.spinner("Analyzing your symptoms..."):
            try:
                result = predict_disease(symptoms)
                st.markdown(f'<div class="result-box">🧬 You may be suffering from: <strong>{result}</strong></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🩺 About This App")
    st.info("""
This app uses a trained AI model to predict diseases from your symptoms.

Built with:
- Python + scikit-learn
- Streamlit frontend
- Local model + encoders

— Made with 🧠 by Mayur
    """)
