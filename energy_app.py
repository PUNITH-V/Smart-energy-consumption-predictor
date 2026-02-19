import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔋 Smart Energy Consumption Predictor")
st.markdown("AI powered household electricity usage forecasting dashboard")

# ================= LOAD =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# ================= SIDEBAR =================
st.sidebar.header("⚙ Input Parameters")

voltage = st.sidebar.number_input("Voltage", value=240.0)
reactive = st.sidebar.number_input("Reactive Power", value=0.1)
sub1 = st.sidebar.number_input("Sub Metering 1", value=0.0)
sub2 = st.sidebar.number_input("Sub Metering 2", value=0.0)
sub3 = st.sidebar.number_input("Sub Metering 3", value=0.0)
hour = st.sidebar.slider("Hour", 0, 23, 12)
weekday = st.sidebar.slider("Weekday", 0, 6, 3)
lag1 = st.sidebar.number_input("Lag 1", value=0.3)
lag24 = st.sidebar.number_input("Lag 24", value=0.3)
rolling = st.sidebar.number_input("Rolling Mean", value=0.3)

# ================= INPUT DATAFRAME =================
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
    "rolling_mean_3": rolling
}])

input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# ================= PREDICTION =================
if st.button("Predict Energy Consumption"):

    scaled = scaler.transform(input_data)
    prediction_scaled = model.predict(scaled)[0]
    prediction = target_scaler.inverse_transform([[prediction_scaled]])[0][0]

    # ================= OUTPUT =================
    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Energy", round(prediction,3))
    col2.metric("Hour", hour)
    col3.metric("Weekday", weekday)

    # ================= STATUS =================
    if prediction > -17:
        st.error("⚠ High Consumption")
        level = "High"
    elif prediction > -18:
        st.warning("⚡ Moderate Usage")
        level = "Medium"
    else:
        st.success("✅ Efficient Usage")
        level = "Low"

    # ================= BAR VISUAL =================
    st.subheader("Prediction Level")

    fig, ax = plt.subplots()
    ax.bar(["Energy"], [prediction])
    ax.set_ylabel("Consumption")
    st.pyplot(fig)

    # ================= FEATURE CONTRIBUTION =================
    st.subheader("Feature Values Visualization")

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.bar(input_data.columns, input_data.iloc[0])
    plt.xticks(rotation=90)
    st.pyplot(fig2)

    # ================= INPUT DISTRIBUTION =================
    st.subheader("Input Comparison")

    avg_values = np.mean(input_data.values)

    fig3, ax3 = plt.subplots()
    ax3.bar(["Your Input","Average"], [np.mean(input_data.values), avg_values])
    st.pyplot(fig3)

    # ================= RADAR STYLE VISUAL =================
    st.subheader("Input Pattern Shape")

    values = input_data.values.flatten()
    angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, polar=True)
    ax4.plot(angles, values)
    ax4.fill(angles, values, alpha=0.1)
    ax4.set_xticks(angles)
    ax4.set_xticklabels(input_data.columns, fontsize=8)
    st.pyplot(fig4)

    # ================= INTERPRETATION =================
    st.subheader("📈 Interpretation")

    st.info(f"""
Prediction Level: **{level}**

Insights:
- Higher voltage or reactive power increases prediction
- Sub-meter values indicate appliance usage impact
- Lag features reflect past consumption behavior
- Rolling mean represents trend pattern
""")

    # ================= RAW DEBUG =================
    with st.expander("See Raw Inputs"):
        st.write(input_data)
