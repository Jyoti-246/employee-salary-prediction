import streamlit as st
import joblib
import numpy as np

# Load saved model and encoder
model = joblib.load("salary_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("💰 Employee Salary Prediction")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

job_title = st.selectbox(
    "Job Title",
    [
        "Software Engineer",
        "Data Analyst",
        "Manager"
    ]
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=1
)

if st.button("Predict Salary"):

    user_data = np.array([
        [age, gender, education, job_title, experience]
    ])

    user_data_encoded = encoder.transform(user_data)

    prediction = model.predict(user_data_encoded)

    st.success(
        f"Predicted Salary: ₹{prediction[0]:,.0f}"
    )