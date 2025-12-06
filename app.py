import streamlit as st
import pandas as pd
from joblib import load
import os

# -------------------------------
# 1️⃣ Load trained model safely
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path=r"heart_failure_prediction_compressed.joblib"):
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    # Load the model
    model = load(model_path)
    return model

model = load_model()
if model is None:
    st.stop()  # Stop if model couldn't be loaded

# -------------------------------
# 2️⃣ Define feature names
# Must exactly match model training columns
# -------------------------------
feature_cols = [
    'age',
    'anaemia',
    'creatinine_phosphokinase',
    'diabetes',
    'ejection_fraction',
    'high_blood_pressure',
    'platelets',
    'serum_creatinine',
    'serum_sodium',
    'sex',
    'smoking',
    'time'
]

st.title("Heart Failure Prediction App")
st.header("Enter Patient Details")

# -------------------------------
# 3️⃣ Collect user input
# -------------------------------
input_data = {}

for feat in feature_cols:
    if feat in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]:
        input_data[feat] = st.selectbox(f"{feat} (0=No, 1=Yes)", [0, 1])
    else:
        # Set reasonable min/max for numeric features
        if feat == "age":
            input_data[feat] = st.number_input(feat, min_value=1, max_value=120, value=50)
        elif feat == "creatinine_phosphokinase":
            input_data[feat] = st.number_input(feat, min_value=0, max_value=10000, value=100)
        elif feat == "ejection_fraction":
            input_data[feat] = st.number_input(feat, min_value=10, max_value=80, value=30)
        elif feat == "platelets":
            input_data[feat] = st.number_input(feat, min_value=50000, max_value=1000000, value=250000)
        elif feat == "serum_creatinine":
            input_data[feat] = st.number_input(feat, min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        elif feat == "serum_sodium":
            input_data[feat] = st.number_input(feat, min_value=100, max_value=150, value=135)
        elif feat == "time":
            input_data[feat] = st.number_input(feat, min_value=1, max_value=300, value=100)
        else:
            input_data[feat] = st.number_input(feat, value=0.0)

input_df = pd.DataFrame([input_data])

# Ensure feature order matches training
input_df = input_df[feature_cols]

# -------------------------------
# 4️⃣ Predict
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of death

    if prediction == 1:
        st.error(f"⚠ High Risk of Death (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✔ Low Risk of Death (Probability: {prediction_proba:.2f})")