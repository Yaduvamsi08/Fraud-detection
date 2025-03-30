import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

st.title("🔍 Credit Card Fraud Detection")
st.write("Enter transaction details below to predict if it's fraudulent or not.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
trans_type = st.selectbox("Transaction Type", ["ATM Withdrawal", "POS Payment", "Online Transfer", "Bill Payment", "Mobile Recharge"])
time_of_tx = st.slider("Time of Transaction (24h format)", 0, 23)
device = st.selectbox("Device Used", ["Desktop", "Mobile", "Tablet", "Unknown"])
location = st.selectbox("Location", ["New York", "San Francisco", "Chicago", "Houston", "Unknown"])
prev_frauds = st.number_input("Previous Fraudulent Transactions", min_value=0)
account_age = st.number_input("Account Age (months)", min_value=0)
tx_last_24h = st.number_input("Transactions in Last 24H", min_value=0)
payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Net Banking", "UPI", "Wallet", "Unknown"])

# Mapping inputs to match label encoding
trans_type_map = {"ATM Withdrawal":0, "POS Payment":4, "Online Transfer":3, "Bill Payment":2, "Mobile Recharge":1}
device_map = {"Desktop":0, "Mobile":1, "Tablet":2, "Unknown":3}
location_map = {"New York":5, "San Francisco":6, "Chicago":1, "Houston":4, "Unknown":7}
payment_method_map = {"Credit Card":0, "Debit Card":1, "Net Banking":3, "UPI":4, "Wallet":5, "Unknown":2}

# Predict
if st.button("Predict"):
    input_data = np.array([[amount,
                            trans_type_map[trans_type],
                            time_of_tx,
                            device_map[device],
                            location_map[location],
                            prev_frauds,
                            account_age,
                            tx_last_24h,
                            payment_method_map[payment_method]]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 Alert: Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
