# 🌤️ Climate Trend Predictor

<p align="left">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Workflow-Apache%20Airflow-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Cloud-AWS%20S3%20%7C%20RDS-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Data%20Pipeline-Kafka%20%7C%20Airflow%20%7C%20S3-lightgrey?style=flat-square" />
</p>

# Climate Trend Predictor

**Climate Trend Predictor** is a production-grade, cloud-native data engineering and forecasting pipeline built with Apache Airflow, Kafka, AWS S3, Amazon RDS, and TensorFlow.

It provides automated ingestion, transformation, feature engineering, model training, and real-time prediction for weather and climate trends.

---

## Key Capabilities
- Automated ETL orchestration with Airflow  
- Real-time & batch ingestion via API and Kafka  
- Scalable AWS S3 data lake (raw → transformed → sequences → models)  
- Data cleaning, validation, and feature engineering  
- LSTM forecasting model in TensorFlow (saved as .keras)  
- End-to-end provenance logging into Amazon RDS  
- Dashboard-ready outputs (Tableau / Plotly)  
- Modular, production-oriented architecture  

---

## Profiles
- **GitHub:** https://github.com/rameyjm7  
- **HuggingFace:** https://huggingface.co/rameyjm7  
- **Kaggle:** https://www.kaggle.com/rameyjm7  

---

# Repository Structure

```
src/
 ├── ingestion/
 │     └── ingest_weather_forecast.py
 ├── transform/
 │     ├── transform_weather_data.py
 │     └── prepare_sequences.py
 ├── training/
 │     └── train_lstm_model.py
 ├── inference/
 │     └── predict_next_hour.py
 ├── airflow/
 │     └── weather_pipeline_dag.py
 ├── utils/
 │     ├── db_utils.py
 │     └── provenance_utils.py
 └── archive/POC/
```

---

# System Overview

The system is composed of four primary layers:

1. **Ingestion Layer**  
   API-driven ingestion + Kafka micro-batch pipelines.

2. **Transformation Layer**  
   Data cleaning, validation, schema enforcement, and feature engineering.

3. **Modeling Layer**  
   LSTM training, saving `.keras` models, and evaluation.

4. **Inference Layer**  
   Automated predictions stored in Amazon RDS for BI dashboards.

---

# 📦 Model Card – Weather LSTM Forecaster

## Model Overview
The **Weather LSTM Forecaster** is a multivariate LSTM model designed to predict short-term temperature trends using cleaned and scaled environmental features.  
It consumes time-series sequences prepared by the pipeline.

---

## Intended Use
- Short-term temperature prediction  
- Climate pattern exploration  
- Tableau / BI dashboards  
- Operational forecasting systems  

### Not Intended For
- Emergency weather alerts  
- Safety‑critical climate predictions  
- Long‑range climate science  

---

## Training Data
The model is trained on:
- Cleaned OpenWeatherMap 5‑day/3‑hour weather forecast data  
- Features:
  - temperature  
  - humidity  
  - wind_speed  
  - precipitation  
- Time series built using **TimeSeriesSplit (10 splits)**  
- Scaling via **MinMaxScaler**

---

## Model Architecture

```
LSTM(32 units, ReLU)
Dense(1)
Loss: MSE
Optimizer: Adam
Epochs: 25
Batch size: 8
```

---

## Evaluation
The training pipeline logs:
- Training MSE  
- Validation MSE  

Tableau dashboards compute:
- MAE  
- RMSE  

---

## Limitations
- Accuracy depends on API data quality  
- Designed for short-term predictions only  
- Not suitable for high-stakes decision-making  

---

## Ethical Considerations
- No personal or private data used  
- Forecasts are statistical and approximate  
- Safe for research, experimentation, and non-critical use  

---

# Author

**Jacob M. Ramey**  
Embedded Systems & Machine Learning Engineer  
GitHub: https://github.com/rameyjm7  
HuggingFace: https://huggingface.co/rameyjm7  
Kaggle: https://www.kaggle.com/jacobramey
