"""
Weather Pipeline DAG
Jacob M. Ramey
ECE5984 Final Project

Daily pipeline for:
1. Weather forecast ingestion
2. Weather historical ingestion
3. Weather live ingestion
4. Sequence preparation
5. LSTM training
6. Multi-step inference
"""

import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Ensure project root in PYTHONPATH for Airflow workers
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ----------------------------------------------------------------------
# Import pipeline modules
# ----------------------------------------------------------------------
from src.ingestion.ingest_weather_forecast import ingest_weather_forecast
from src.ingestion.ingest_weather_historical import ingest_weather_historical
from src.ingestion.ingest_weather_live import ingest_weather_live
from src.transform.prepare_sequences import prepare_sequences
from src.training.train_lstm_model import train_lstm_weather
from src.inference.predict_weather import predict_next_n_hours

# ----------------------------------------------------------------------
# Default Airflow args
# ----------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["rameyjm7@vt.edu"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 11, 1),
}

# ----------------------------------------------------------------------
# DAG definition
# ----------------------------------------------------------------------
dag = DAG(
    dag_id="weather_full_pipeline",
    default_args=default_args,
    description="Daily ETL, sequence generation, model training, and inference pipeline",
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# ----------------------------------------------------------------------
# Task definitions
# ----------------------------------------------------------------------

task_ingest_forecast = PythonOperator(
    task_id="ingest_weather_forecast",
    python_callable=ingest_weather_forecast,
    dag=dag,
)

task_ingest_historical = PythonOperator(
    task_id="ingest_weather_historical",
    python_callable=ingest_weather_historical,
    dag=dag,
)

task_ingest_live = PythonOperator(
    task_id="ingest_weather_live",
    python_callable=ingest_weather_live,
    dag=dag,
)

task_prepare_sequences = PythonOperator(
    task_id="prepare_sequences",
    python_callable=prepare_sequences,
    dag=dag,
)

task_train_model = PythonOperator(
    task_id="train_lstm_model",
    python_callable=train_lstm_weather,
    dag=dag,
)

task_run_inference = PythonOperator(
    task_id="run_inference",
    python_callable=lambda: predict_next_n_hours(steps=8),
    dag=dag,
)

# ----------------------------------------------------------------------
# Pipeline dependency chain
# ----------------------------------------------------------------------

(
    task_ingest_forecast
    >> task_ingest_historical
    >> task_ingest_live
    >> task_prepare_sequences
    >> task_train_model
    >> task_run_inference
)
