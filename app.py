pip install joblib


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sales Forecasting", layout="centered")
st.title("📊 Sales Forecasting App")

# Load Model
try:
    model = joblib.load("lightgbm_model.pkl")
except:
    st.error("Model file not found!")
    st.stop()

st.success("Model Loaded Successfully ✅")

# User Inputs (ONLY 7 FEATURES)

store_id = st.number_input("StoreID_encoded", min_value=0)
product_id = st.number_input("ProductID_encoded", min_value=0)

lag1 = st.number_input("Lag1 (Yesterday Sales)", min_value=0.0)
lag7 = st.number_input("Lag7 (Last Week Same Day Sales)", min_value=0.0)

date = st.date_input("Select Date")

# Feature Engineering
day = date.day
month = date.month
weekday = date.weekday()

# Create Input DataFrame
input_df = pd.DataFrame([[
    store_id,
    product_id,
    lag1,
    lag7,
    day,
    month,
    weekday
]], columns=[
    'StoreID_encoded',
    'ProductID_encoded',
    'Lag1',
    'Lag7',
    'Day',
    'Month',
    'Weekday'
])

# Prediction
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Units Sold: {round(prediction,2)}")
