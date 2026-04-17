import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("../model.pkl")

# Title
st.title("Medical Cost Prediction App")

st.write("Enter patient details to predict insurance cost")

# Inputs
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Predict button
if st.button("Predict Cost"):
    data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(data)[0]

    st.success(f"💰 Estimated Medical Cost: ₹{round(prediction, 2)}")