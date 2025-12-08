✈️ Aircraft Engine Predictive Maintenance
Predicting Remaining Useful Life (RUL) Using NASA CMAPSS Turbofan Data

This project builds a real-world predictive maintenance system using the NASA C-MAPSS turbofan dataset.
The goal is to predict Remaining Useful Life (RUL) of jet engines using deep learning models (LSTM & GRU).

🚀 Project Summary (Short)

Built end-to-end predictive maintenance pipeline

Prepared time-series sensor sequences for modeling

Trained LSTM and GRU models for accurate RUL prediction

Saved best models for real-time inference

Preparing a Streamlit maintenance dashboard (Week 4)
```
Project Structure
notebooks/
    01_EDA.ipynb
    02_Feature_Engineering.ipynb
   03_Model_Training_v2.ipynb
models/
    lstm_fd001_best.h5
    gru_fd001_best.h5
data/
    raw/
    processed/
    ```
 Weekly Progress (Concise)
 Week 1: EDA

Loaded NASA CMAPSS dataset

Visualized engine degradation & sensor behavior

Identified key sensors and failure cycles

 Week 2: Feature Engineering

Normalized selected sensors

Created sliding windows (sequence length = 30)

Generated RUL labels

Saved training arrays (X_train, y_train)

Week 3: Model Training (LSTM & GRU)

Trained two deep learning models:

Model	MAE	RMSE
LSTM	x.xx	x.xx
GRU	x.xx	x.xx

Replace x.xx with your actual values.

Saved models for deployment:

models/lstm_fd001_best.h5
models/gru_fd001_best.h5

 Week 4 (Next): Streamlit Dashboard

Real-time RUL prediction

Engine health status

Interactive sensor visualization

 Installation
pip install -r requirements.txt

Run notebooks:
jupyter notebook

📚 Dataset

NASA C-MAPSS Turbofan Engine Degradation Dataset
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

👤 Author

Goitom Abirha
Data Scientist – Predictive Maintenance | Deep Learning


