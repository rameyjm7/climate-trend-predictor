# Jacob M. Ramey
# ECE5984, Homework 2
# this is used by Apache Airflow to launch our code
from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from batch_ingest import ingest_weather_data
from transform import transform_weather_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 9, 7),
    'email': ['rameyjm7@vt.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'batch_ingest_dag',
    default_args=default_args,
    description='ingest weather data',
    schedule_interval=timedelta(days=1),
)

ingest_etl = PythonOperator(
    task_id='ingest_weather_dataset',
    python_callable=ingest_weather_data,
    dag=dag,
)

transform_etl = PythonOperator(
    task_id='transform_weather_dataset',
    python_callable=transform_weather_data,
    dag=dag,
)

ingest_etl >> transform_etl

# ingest_etl