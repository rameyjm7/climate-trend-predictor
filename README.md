# 🌤️ Climate Trend Predictor

<p align="left">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Workflow-Apache%20Airflow-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Cloud-AWS%20S3%20%7C%20RDS-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Data%20Pipeline-Python%20%7C%20Airflow-lightgrey?style=flat-square" />
</p>

# Climate Trend Predictor

**Climate Trend Predictor** is an end-to-end data engineering pipeline for city-level weather forecasting.  
It automates ingestion, cleaning, transformation, LSTM model training, inference, and visualization using a Batch–ML–Visualization pipeline deployed both locally and on AWS.

---

# Profiles
- **GitHub:** https://github.com/rameyjm7  
- **HuggingFace:** https://huggingface.co/rameyjm7  
- **Kaggle:** https://www.kaggle.com/rameyjm7  

---

# 1. Title
**Climate Trend Predictor: A Batch–ML–Visualization Weather Forecasting System**

---

# 2. Project Function
This pipeline forecasts temperature trends using cleaned historical, live, and forecast weather data.  
It ingests data daily, maintains provenance, generates sliding-window sequences, trains an LSTM, and performs multi-hour inference (+8h).  
Plots and BI-ready visualizations are automatically generated.

---

# 3. Dataset
Data is sourced from OpenWeatherMap (forecast + live) and custom historical scraping to increase coverage.  
Features include temperature, humidity, wind speed, and precipitation.  
Merged and cleaned data is stored in local/S3 dual-mode folders or in MySQL RDS.

---

# 4. Pipeline / Architecture
This project follows the **Batch-ML-Visualization pipeline**:

- **Ingestion:** Python scripts + OpenWeatherMap API  
- **Storage:** Local filesystem + AWS S3 + MySQL RDS  
- **Transformation:** Cleaning, normalization, time-windowing  
- **Modeling:** TensorFlow LSTM forecaster trained for 150 epochs  
- **Orchestration:** Apache Airflow DAG running daily  
- **Visualization:** Matplotlib outputs for BI dashboards  

<infographic>
Pipeline Architecture Diagram Placeholder

<photo about X>
Example Visualization Placeholder

---

# 5. Data Quality Assessment
The dataset required substantial cleaning due to missing timestamps, inconsistent forecast intervals, and sparse values.  
Data quality improved dramatically after scraping historical weather to fill coverage gaps.  
Normalization, deduplication, type enforcement, and outlier checks were applied across all ingestion sources.

---

# 6. Data Transformations & Models Used
Transformations include merging datasets, feature standardization, and building 40-hour sliding windows.  
A stacked LSTM model (128/64 units) with dropout and dense layers was trained for 150 epochs.  
Accuracy improved significantly after increasing dataset size and training duration.  
Inference uses autoregressive multi-step prediction (+8 hours).

<photo about model results>
Forecast Results Visualization Placeholder

---

# 7. Infographic
<infographic>
System Overview Infographic Placeholder

---

# 8. Code Repository
GitHub Repository:  
https://github.com/your-username/climate-trend-predictor

---

# 9. Thorough Investigation

### What Worked
- The LSTM RNN effectively learned temporal dependencies once sufficient data was provided.  
- Scraping additional historical data greatly improved performance and reduced MAE.  
- Increasing training epochs to 150 (from 20) stabilized convergence and eliminated underfitting.  
- Combining historical, live, and forecast data expanded the time horizon and improved model generalization.

### What Did Not Work
- The initial dataset was too limited to capture meaningful temperature patterns, resulting in poor accuracy.  
- Early training attempts underfit due to insufficient temporal depth.  
- Forecast-only data lacked variability and did not represent real atmospheric dynamics well.  
- The model struggled with abrupt weather changes due to limited input features.

### Model Complexity and Real-World Considerations
In operational meteorology, weather prediction is performed using highly complex physics-based numerical weather prediction (NWP) systems such as HRRR, GFS, and ECMWF.  
These models incorporate dozens of atmospheric variables, satellite and radar observations, pressure fields, humidity layers, wind vectors at multiple altitudes, and fluid dynamics equations.  

By comparison, this project uses a **minimal feature set (temperature, humidity, wind speed, precipitation)** and a relatively small LSTM architecture.  
Despite these constraints, the model performs **surprisingly well**, demonstrating that even lightweight data-driven models can capture short-term temporal trends.  
Performance would improve further with:
- Additional atmospheric features (pressure, cloud cover, dew point, visibility)  
- Multi-layer radar reflectivity  
- Satellite infrared/visible channels  
- Longer historical time series  
- Spatial features (neighboring city conditions)

### Scalability, Viability, and Recommendations
The pipeline’s modular design makes it viable for production-scale forecasting with more data sources and expanded feature engineering.  
Airflow orchestration, dual AWS/local storage, and clear separation of ingestion, transformation, modeling, and inference support future scalability.  
Recommended next steps include building a multivariate LSTM or Transformer, integrating radar/satellite inputs, and deploying the predictions to a real-time dashboard or API endpoint.

---

# 10. Running the Project

### Manual Pipeline Execution
