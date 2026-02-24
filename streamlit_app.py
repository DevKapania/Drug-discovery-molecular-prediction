"""
Efficient Drug Discovery - Streamlit Web App
Author: Dev Kapania | IIT Roorkee Research Intern

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Drug Discovery Predictor",
    page_icon="💊",
    layout="wide"
)

# ── Load Model & Scaler ────────────────────────────────
@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model  = keras.models.load_model(os.path.join(models_dir, 'drug_dnn_final.h5'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    return model, scaler

# ── UI ─────────────────────────────────────────────────
st.title("💊 Drug Candidate Activity Predictor")
st.markdown("**Predict whether a drug candidate is biologically active or inactive using Deep Learning.**")
st.markdown("---")

st.sidebar.header("📋 About")
st.sidebar.info(
    "This app uses a Deep Neural Network trained on molecular descriptors "
    "to classify drug candidates as **Active** or **Inactive**.\n\n"
    "**Model:** DNN (256→128→64→1)\n"
    "**Author:** Dev Kapania\n"
    "**Institution:** IIT Roorkee"
)

st.subheader("Enter Molecular Descriptors")
st.markdown("Input the key physicochemical properties of your compound:")

col1, col2, col3 = st.columns(3)

with col1:
    mol_weight   = st.number_input("Molecular Weight (g/mol)", min_value=0.0, value=250.0, step=1.0)
    logp         = st.number_input("LogP (lipophilicity)", min_value=-10.0, max_value=10.0, value=2.5)
    hbd          = st.number_input("H-Bond Donors", min_value=0, max_value=20, value=2)

with col2:
    hba          = st.number_input("H-Bond Acceptors", min_value=0, max_value=20, value=4)
    tpsa         = st.number_input("TPSA (Å²)", min_value=0.0, max_value=300.0, value=60.0)
    rot_bonds    = st.number_input("Rotatable Bonds", min_value=0, max_value=20, value=4)

with col3:
    arom_rings   = st.number_input("Aromatic Rings", min_value=0, max_value=10, value=2)
    heavy_atoms  = st.number_input("Heavy Atom Count", min_value=0, max_value=100, value=20)
    frac_csp3    = st.number_input("Fraction Csp3", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

st.markdown("---")

if st.button("🔬 Predict Drug Activity", use_container_width=True):
    try:
        model, scaler = load_artifacts()
        features = np.array([[mol_weight, logp, hbd, hba, tpsa,
                               rot_bonds, arom_rings, heavy_atoms, frac_csp3]])

        # Pad to expected input size if needed
        expected = model.input_shape[1]
        if features.shape[1] < expected:
            padding = np.zeros((1, expected - features.shape[1]))
            features = np.hstack([features, padding])

        features_scaled = scaler.transform(features)
        prob = model.predict(features_scaled, verbose=0).flatten()[0]
        label = "🟢 ACTIVE" if prob >= 0.5 else "🔴 INACTIVE"

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Prediction", label)
        with col_r2:
            st.metric("Confidence Score", f"{prob*100:.1f}%")

        st.progress(float(prob))

        if prob >= 0.5:
            st.success(f"This compound is predicted to be **biologically active** with {prob*100:.1f}% confidence.")
        else:
            st.warning(f"This compound is predicted to be **inactive** with {(1-prob)*100:.1f}% confidence.")

    except Exception as e:
        st.error(f"Error: {e}\n\nPlease run the training pipeline first to generate model files.")
        st.info("Run: `python src/train.py` to train the model.")

st.markdown("---")
st.caption("Built by Dev Kapania | Deep Learning Research Intern @ IIT Roorkee | github.com/DevKapania")
