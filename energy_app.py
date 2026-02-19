import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import time

st.set_page_config(layout="wide")

# ================= THEME TOGGLE =================
theme = st.sidebar.selectbox("Theme", ["Dark","Light"])

if theme == "Dark":
    st.markdown("""
    <style>
    body {background-color:#0E1117; color:white;}
    </style>
    """, unsafe_allow_html=True)

# ================= LOAD =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# ================= HEADER =================
st.title("🔋 Smart Energy Consumption AI Dashboard")
st.caption("Real-time household electricity prediction system")

# ================= INPUT PANEL =================
st.sidebar.header("Input Parameters")

voltage = st.sidebar.number_input("Voltage", 200.0,260.0,240.0)
reactive = st.sidebar.slider("Reactive Power",0.0,1.0,0.1)
sub1 = st.sidebar.slider("Sub Metering 1",0.0,5.0,0.0)
sub2 = st.sidebar.slider("Sub Metering 2",0.0,5.0,0.0)
sub3 = st.sidebar.slider("Sub Metering 3",0.0,5.0,0.0)
hour = st.sidebar.slider("Hour",0,23,12)
weekday = st.sidebar.slider("Weekday",0,6,3)
lag1 = st.sidebar.slider("Lag 1",0.0,5.0,0.3)
lag24 = st.sidebar.slider("Lag 24",0.0,5.0,0.3)
rolling = st.sidebar.slider("Rolling Mean",0.0,5.0,0.3)

# ================= INPUT DF =================
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

# ================= REALTIME SIMULATION =================
auto = st.sidebar.checkbox("Live Prediction Mode")

def predict():
    scaled = scaler.transform(input_data)
    pred_scaled = model.predict(scaled)[0]
    pred = target_scaler.inverse_transform([[pred_scaled]])[0][0]
    return pred

if auto:
    placeholder = st.empty()
    for _ in range(1000):
        prediction = predict()
        placeholder.metric("Live Energy Prediction", round(prediction,3))
        time.sleep(1)
else:
    if st.button("Predict"):
        prediction = predict()
    else:
        prediction = None

# ================= OUTPUT =================
if prediction is not None:

    st.divider()
    st.subheader("Prediction Dashboard")

    col1,col2,col3 = st.columns(3)

    col1.metric("Predicted Energy", round(prediction,3))
    col2.metric("Hour",hour)
    col3.metric("Weekday",weekday)

    # ================= STATUS =================
    if prediction > -17:
        level="High"
        st.error("High Consumption Detected")
    elif prediction > -18:
        level="Moderate"
        st.warning("Moderate Usage")
    else:
        level="Efficient"
        st.success("Efficient Usage")

    # ================= GAUGE =================
    st.subheader("Consumption Meter")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        gauge={
            "axis":{"range":[-20,0]},
            "steps":[
                {"range":[-20,-18],"color":"green"},
                {"range":[-18,-17],"color":"orange"},
                {"range":[-17,0],"color":"red"}
            ]
        }
    ))
    st.plotly_chart(gauge,use_container_width=True)

    # ================= FEATURE BAR =================
    st.subheader("Feature Contribution")

    fig = px.bar(
        x=input_data.columns,
        y=input_data.iloc[0],
        title="Input Feature Values"
    )
    st.plotly_chart(fig,use_container_width=True)

    # ================= PIE =================
    st.subheader("Input Distribution")

    pie = px.pie(
        names=input_data.columns,
        values=input_data.iloc[0]
    )
    st.plotly_chart(pie,use_container_width=True)

    # ================= SHAP =================
    st.subheader("Model Explainability (SHAP)")

    explainer = shap.Explainer(model, scaler.transform(input_data))
    shap_values = explainer(scaler.transform(input_data))

    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact": shap_values.values[0]
    })

    shap_fig = px.bar(
        shap_df.sort_values("Impact"),
        x="Impact",
        y="Feature",
        orientation="h",
        title="Feature Impact on Prediction"
    )
    st.plotly_chart(shap_fig,use_container_width=True)

    # ================= AI RECOMMENDATIONS =================
    st.subheader("AI Recommendations")

    tips=[]

    if voltage > 245:
        tips.append("Reduce voltage fluctuation devices")

    if sub1+sub2+sub3 > 3:
        tips.append("Multiple appliances running simultaneously")

    if hour > 18:
        tips.append("Peak hour usage detected")

    if lag24 > 1:
        tips.append("Yesterday usage was high")

    if not tips:
        tips.append("Energy usage looks optimal")

    for t in tips:
        st.info(t)

    # ================= ANIMATED TREND =================
    st.subheader("Prediction Trend Simulation")

    chart = st.line_chart(np.zeros(20))

    data = np.zeros(20)

    for i in range(20):
        data[i] = prediction + np.random.normal(0,0.2)
        chart.line_chart(data)
        time.sleep(0.05)

    # ================= RAW =================
    with st.expander("View Raw Input"):
        st.write(input_data)
