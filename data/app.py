import streamlit as st
import pandas as pd
import joblib
st.set_page_config(
    page_title="AI-Powered Diabetes Classification",
    page_icon="🩺",
    layout="wide"
)
# Load trained model
model = joblib.load("../model/diabetes_model.pkl")
with st.sidebar:
    st.header("📊 Project Information")

    st.write("""
**Project Name**
AI-Powered Diabetes Classification System

**Algorithm**
Random Forest Classifier

**Technology**
- Python
- Pandas
- Scikit-learn
- Streamlit

**Purpose**
             **Model Accuracy**
72.08%

**Dataset Records**
768

**Features**
8

**Prediction Type**
Binary Classification
Predict whether a patient is likely to have diabetes based on medical data.
""")
st.title("AI-Powered Diabetes Classification System")
st.markdown("""
### Welcome 👋

This application predicts whether a patient is likely to have diabetes using a Machine Learning model.

Please enter the patient's details below and click **Predict**.
""")
st.write("Enter the patient details below:")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)


if st.button("Predict"):
    data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    if prediction[0] == 1:
        st.error("⚠️ Diabetes Detected")
        st.progress(int(probability[0][1] * 100))
        st.write(f"Risk Probability: **{probability[0][1] * 100:.2f}%**")
    else:
        st.success("✅ No Diabetes Detected")
        st.progress(int(probability[0][0] * 100))
        st.write(f"Confidence: **{probability[0][0] * 100:.2f}%**")