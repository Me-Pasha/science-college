import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib 
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("log_reg_diabetes_model.pkl")

# Title and description
st.title("Diabetes Prediction App")
st.write("""
This app predicts whether a person is **diabetic** based on medical parameters using a **Logistic Regression** model.
""")

# Input form
st.header("Enter Patient Data:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]]) 

    scaler = StandardScaler()
    x = scaler.fit_transform(input_data)
        
    prediction = model.predict(x)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    st.subheader("Prediction Result:")
    st.success(f"The patient is **{result}**.")

