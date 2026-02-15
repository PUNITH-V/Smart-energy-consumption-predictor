import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("🔋 Smart Energy Consumption Predictor")
st.write("Predict household energy usage using trained ML model.")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.header("Input Features")

voltage = st.sidebar.number_input("Voltage", value=240.0)
reactive = st.sidebar.number_input("Global Reactive Power", value=0.1)
sub1 = st.sidebar.number_input("Sub Metering 1", value=0.0)
sub2 = st.sidebar.number_input("Sub Metering 2", value=0.0)
sub3 = st.sidebar.number_input("Sub Metering 3", value=0.0)
hour = st.sidebar.slider("Hour", 0, 23, 12)
weekday = st.sidebar.slider("Weekday (0=Mon)", 0, 6, 3)
lag1 = st.sidebar.number_input("Lag 1", value=0.3)
lag24 = st.sidebar.number_input("Lag 24", value=0.3)
rolling = st.sidebar.number_input("Rolling Mean 24", value=0.3)

input_data = pd.DataFrame([{
    "Voltage": voltage,
    "Global_reactive_power": reactive,
    "Sub_metering_1": sub1,
    "Sub_metering_2": sub2,
    "Sub_metering_3": sub3,
    "hour": hour,
    "weekday": weekday,
    "lag_1": lag1,
    "lag_24": lag24,
    "rolling_mean_3": rolling   # ← FIXED
}])


input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)


if st.button("Predict Energy Consumption"):

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]

    st.subheader("📊 Prediction Result")

    # Show raw number clearly
    st.write("Raw prediction value:", float(prediction))

    st.write("Debug – All inputs:")
    st.write(input_data)

    if prediction > 0.30:
        st.warning("⚠ High consumption detected!")
    elif prediction > 0.20:
        st.info("⚡ Moderate usage")
    else:
        st.success("✅ Efficient energy usage")



    st.bar_chart([prediction])
