# 🔋 Smart Energy Consumption Predictor

A machine learning-based energy consumption forecasting system that predicts household energy usage using time-series feature engineering and a trained ML model. The project includes an interactive Streamlit dashboard for real-time predictions.

---

## 📌 Project Overview

This project aims to predict household energy consumption based on electrical and time-based features. 

It includes:

- Data preprocessing and feature engineering
- Baseline model comparison
- LSTM-based time-series modeling
- Model evaluation using MAE and RMSE
- Deployment using Streamlit Cloud

---

## 🧠 Features Used

The model uses the following input features:

- Voltage  
- Global Reactive Power  
- Sub Metering 1  
- Sub Metering 2  
- Sub Metering 3  
- Hour  
- Weekday  
- Lag 1  
- Lag 24  
- Rolling Mean (3-hour window)  

---

## 📊 Model Evaluation

Performance metrics used:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

The model performance was compared against a baseline regression model to evaluate prediction improvement.

---

## 🖥️ Deployment

The model is deployed using **Streamlit** to provide an interactive dashboard where users can:

- Input energy parameters
- Get real-time predictions
- View energy consumption insights

---

## 🚀 How to Run Locally

1. Clone the repository:

```
git clone https://github.com/your-username/smart-energy-consumption-predictor.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run energy_app.py
```

---

## 📂 Project Structure

```
smart-energy-consumption-predictor/
│
├── energy_app.py      # Streamlit dashboard
├── model.pkl          # Trained ML model
├── scaler.pkl         # Feature scaler
├── requirements.txt   # Dependencies
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LSTM (TensorFlow / Keras)
- Streamlit

---

## 🎯 Future Improvements

- Improve LSTM performance with hyperparameter tuning
- Add real-time data integration
- Deploy using Docker for production scalability
- Add historical trend visualization

---

## 👨‍💻 Author

Developed as part of a Machine Learning project focused on time-series energy consumption forecasting and deployment.

