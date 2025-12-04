# Jacob M. Ramey
# ECE5984, Final Project
# Airflow DAG for daily weather data ingestion and transformation

from ingest_weather_forecast import ingest_weather_forecast
from transform_weather_data import transform_weather_data
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
dag = DAG(
    "weather_pipeline_dag",
    default_args=default_args,
    description="Daily pipeline for weather data ingestion and transformation",
    schedule_interval=timedelta(days=1),  # runs once per day
    catchup=False,
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
ingest_task = PythonOperator(
    task_id="ingest_weather_data",
    python_callable=ingest_weather_forecast,
    dag=dag,
)

transform_task = PythonOperator(
    task_id="transform_weather_data",
    python_callable=transform_weather_data,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Workflow dependency
# ---------------------------------------------------------------------------
ingest_task >> transform_task