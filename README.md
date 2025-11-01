# Climate Trend Predictor

**Climate Trend Predictor** is an end-to-end data engineering pipeline for weather forecasting built with Apache Kafka, Apache Airflow, AWS S3, and TensorFlow.
It ingests real-time and historical data from the OpenWeatherMap API, performs batch and micro-batch transformations, and prepares the dataset for machine-learning-based climate prediction.

## Key Features
- Automated ETL pipeline for weather and climate data
- Kafka-based ingestion orchestrated by Apache Airflow
- Scalable AWS S3 data lake and transformation layer
- Data cleaning, outlier removal, and validation using pandas
- LSTM forecasting model implemented in TensorFlow / Keras
- Visualization-ready outputs for Tableau or Plotly dashboards
- Modular proof of concept under src/POC/ for reproducible experimentation

## Tech Stack
```
| Layer                           | Technologies                              |
|--------------------------------|-------------------------------------------|
| Programming                    | Python, pandas, NumPy                     |
| Ingestion                      | OpenWeatherMap API, Apache Kafka          |
| Workflow Orchestration         | Apache Airflow                            |
| Data Storage                   | AWS S3, Amazon RDS                        |
| Compute                        | AWS EC2, local simulation                 |
| Machine Learning               | TensorFlow, Keras                         |
| Visualization                  | Tableau, Plotly                           |
| Version Control / DevOps       | Git, GitHub                               |
```

## Repository Structure
```
src/
 ├── POC/
 │    ├── batch_ingest.py       # Collects and stores raw weather data
 │    ├── transform.py          # Cleans, validates, and aggregates datasets
 │    ├── EDA.py                # Exploratory data analysis and feature inspection
 │    └── dag.py                # Airflow DAG for automated execution
 └── (future modules to be added here)
```

Future phases will expand src/ with production-ready components for model training, evaluation, RDS integration, and dashboard deployment.

## Project Overview
This project demonstrates a complete data engineering workflow — from raw data ingestion to machine learning-driven forecasting.
Developed for ECE 5984: Data Engineering at Virginia Tech, it highlights scalable, cloud-integrated, and ML-ready data pipelines built with modern tools like Kafka, Airflow, and AWS.

## Author
Jacob M. Ramey
Virginia Tech – Department of Electrical and Computer Engineering
