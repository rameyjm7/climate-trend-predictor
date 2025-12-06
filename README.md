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

To get a free API key from OpenWeatherMap, go here https://openweathermap.org/api
1000 API calls a day is free

---

# 4. Pipeline / Architecture
This project follows the **Batch-ML-Visualization pipeline**:

- **Ingestion:** Python scripts + OpenWeatherMap API  
- **Storage:** Local filesystem + AWS S3 + MySQL RDS  
- **Transformation:** Cleaning, normalization, time-windowing  
- **Modeling:** TensorFlow LSTM forecaster trained for 150 epochs  
- **Orchestration:** Apache Airflow DAG running daily  
- **Visualization:** Matplotlib outputs for BI dashboards  

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

<img width="751" height="307" alt="image" src="https://github.com/user-attachments/assets/299fb650-469c-40f2-aacb-f7f32a323ace" /> 

Time-series temperature plot for Chicago 


<img width="751" height="307" alt="image" src="https://github.com/user-attachments/assets/d7b7dd3f-fe8e-4802-834c-348bea39c898" />

Figures 4. Time-series humidity plot for Los Angeles.


<img width="416" height="353" alt="image" src="https://github.com/user-attachments/assets/3fb23608-b68e-4e1c-b325-1b654c98a839" />

Feature correlation matrix for meteorological variables.


<img width="454" height="472" alt="image" src="https://github.com/user-attachments/assets/f5b0eb50-14fa-4004-b6a8-6b660e722eb1" />

Pairwise relationship between meteorological variables.


<img width="750" height="293" alt="image" src="https://github.com/user-attachments/assets/fc2ebda5-cbe5-40e4-be86-25139c77a326" />

Temperature autocorrelation confirms daily periodicity.


<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/ae62f5a4-3e97-4735-aca6-ecd02cade0c8" />

Actual versus predicted weather for Seattle


<img width="1268" height="817" alt="image" src="https://github.com/user-attachments/assets/3f3d2573-ad5a-42ab-8621-511a4767c026" />

Training versus validation MSE

---

# 7. Infographic

<img width="512" height="776" alt="image" src="https://github.com/user-attachments/assets/67434b73-d24f-4d63-a4fe-1c903707f9e5" />


---

# 8. Code Repository
GitHub Repository:  
[https://github.com/your-username/climate-trend-predictor](https://github.com/rameyjm7/climate-trend-predictor)

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

In AWS EC2 container:
- spin up the docker container
- cd into the repo root
- run 'pip install -r requirements.txt'
- run 'source env.txt'
- edit src/config.py and put the password where it says RDS_PASS for the default
- cd to the airflow directory
- mkdir dags/
- run 'ln -s /path/to/climate-trend-predictor/src/airflow/weather_pipeline_dag.py dags/'
- start airflow 'airflow standalone'
- log in with the admin password
- run the 'weather_full_pipeline' dag
- view the graph to watch it execute

You can also run the scripts/test_pipeline.sh script inside the docker container to run the pipeline without Airflow, still using AWS S3, EC2, RDS


<img width="975" height="252" alt="image" src="https://github.com/user-attachments/assets/93436c36-d51b-46e3-bb87-ab97ec9415cb" />


DAG running in Apache Airflow


<img width="975" height="252" alt="image" src="https://github.com/user-attachments/assets/5a98607f-9ed0-4297-8579-b6dda5cda71b" />

DAG Pipeline graph


Results are saved to a database and queryable, shown above. Also you can run inference on the CLI and here is an example output

<img width="975" height="342" alt="image" src="https://github.com/user-attachments/assets/058abef8-0dd1-4edc-926a-bdd844e21994" />



