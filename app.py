import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("churn_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on account and service details.")

# -----------------------------
# Sidebar Inputs (Simplified & Grouped)
# -----------------------------
st.sidebar.header("Customer Information")

def user_input():
    # Demographics
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    senior = st.sidebar.checkbox("Senior Citizen")
    partner = st.sidebar.radio("Partner?", ["Yes", "No"])
    dependents = st.sidebar.radio("Dependents?", ["Yes", "No"])
    
    # Account info
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.radio("Paperless Billing?", ["Yes", "No"])
    payment = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    # Services
    phone = st.sidebar.radio("Phone Service?", ["Yes", "No"])
    multiple = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.sidebar.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    online_security = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    # Charges
    monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0)
    total = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 2500.0)

    # Convert checkbox to 0/1
    senior = int(senior)

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    return pd.DataFrame([data])

# -----------------------------
# Get User Input
# -----------------------------
input_df = user_input()

st.subheader("Customer Input Data")
st.write(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn\n\nChurn Probability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is not likely to churn\n\nChurn Probability: {probability:.2%}")
