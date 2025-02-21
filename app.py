import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('churn_model.pkl')

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Enter customer details below to predict churn.")

# User input fields
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
total_charges = st.number_input("Total Charges", min_value=0.0, format="%.2f")
tenure = st.number_input("Tenure (Months)", min_value=0, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

# Convert categorical inputs to one-hot encoding
input_data = pd.DataFrame([{
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "tenure": tenure,
    "gender_Male": 1 if gender == "Male" else 0,
    "gender_Female": 1 if gender == "Female" else 0,
    "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "Dependents_No": 1 if dependents == "No" else 0,
    "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
    "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
    "DeviceProtection_No": 1 if device_protection == "No" else 0
}])

# Ensure input data matches model's expected features
expected_features = model.feature_names_in_
input_data = input_data.reindex(columns=expected_features, fill_value=0)

# Predict churn when user clicks button
if st.button("Predict Churn"):
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        if prediction == 1:
            st.error("🚨 This customer is likely to churn!")
        else:
            st.success("✅ This customer is not likely to churn!")

    except Exception as e:
        st.error(f"Error: {e}")
