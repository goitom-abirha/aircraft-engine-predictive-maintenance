✈️ Aircraft Engine Predictive Maintenance
Predict Remaining Useful Life (RUL) of Jet Engines Using NASA CMAPSS FD001 + LSTM/GRU Models

This project builds a fully operational Predictive Maintenance system using NASA’s C-MAPSS turbofan engine dataset.
It uses deep learning (LSTM + GRU) to estimate Remaining Useful Life (RUL) and includes an interactive Streamlit dashboard for real-time fleet monitoring.

🚀 Key Features
🔧 End-to-End Predictive Maintenance Pipeline

Load & clean NASA CMAPSS FD001 dataset

Create sensor-based time-series sequences

Train deep learning models for RUL regression

Save best models for real-time inference

📊 Streamlit Dashboard (Production-Ready)

Upload custom engine cycles or use built-in sample data

Predict RUL using LSTM or GRU

Generate PDF health reports

SHAP-based or fallback feature importance

Sensor anomaly detection using Z-score

Fleet-level risk ranking

🤖 Deep Learning Models

Sequence length: 30 cycles

Features: key sensors + operational settings

Models: LSTM + GRU

🧠 Model Performance
Model	MAE	RMSE
LSTM	x.xx	x.xx
GRU	x.xx	x.xx

Replace with your actual metrics from Week 3 training.

🗂️ Project Structure
aircraft-engine-predictive-maintenance/
│
├── app.py                     <- Streamlit dashboard
├── requirements.txt
├── README.md
│
├── models/                    <- Saved deep learning models
│     ├── lstm_fd001_best.h5
│     └── gru_fd001_best.h5
│
├── data/
│     └── raw/
│          ├── test_FD001.txt
│          ├── train_FD001.txt
│          └── (other CMAPSS files optional)
│
└── notebooks/
      ├── 01_EDA.ipynb
      ├── 02_Feature_Engineering.ipynb
      └── 03_Model_Training_v2.ipynb

📈 Weekly Progress Summary
Week 1 – Exploratory Data Analysis

Loaded all NASA CMAPSS FD001 files

Visualized sensor degradation

Identified most predictive sensors

Understood operational settings behavior

Week 2 – Feature Engineering

Scaled key sensors using MinMax

Created sliding windows (30-cycle sequences)

Generated RUL labels

Saved training arrays (X_train, y_train)

Week 3 – Model Training

Trained and evaluated two deep learning models:

LSTM

GRU

Saved best model weights:

models/
   lstm_fd001_best.h5
   gru_fd001_best.h5

Week 4 – Streamlit Deployment

Real-time RUL prediction

Engine-level PDF reports

Fleet-level monitoring dashboard

Sensor anomaly detection

SHAP explainability (with safe fallback)

▶️ Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Run the Streamlit dashboard
streamlit run app.py

🌐 Live Application

Replace with your Streamlit Cloud URL:

👉 Live App:
https://your-app-name.streamlit.app

📚 Dataset

NASA C-MAPSS Turbofan Engine Degradation Dataset
🔗 https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

👤 Author – Goitom Abirha

Data Scientist – Predictive Maintenance & Deep Learning
LinkedIn: https://linkedin.com/in/
...
GitHub: https://github.com/goitom-abirha
