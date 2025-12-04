from sqlalchemy import create_engine
import os
from sqlalchemy import Table, Column, Float, String, DateTime, MetaData, Integer

def get_engine(env="local"):
    """
    Returns a SQLAlchemy engine.
    - local: uses SQLite for testing
    - prod: connects to Amazon RDS PostgreSQL
    """
    if env == "local":
        db_path = os.path.abspath("/root/workspace/climate-trend-predictor/data/weather_local.db")
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
    elif env == "prod":
        # Example RDS config (to be updated when deployed)
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        dbname = os.getenv("DB_NAME", "weather_db")
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
    else:
        raise ValueError("Invalid environment specified.")
    return engine


def create_tables(engine):
    metadata = MetaData()

    weather_raw = Table("weather_raw", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("city", String(50)),
        Column("datetime", DateTime),
        Column("temperature", Float),
        Column("temperature_min", Float),
        Column("temperature_max", Float),
        Column("humidity", Float),
        Column("pressure", Float),
        Column("wind_speed", Float),
        Column("precipitation", Float),
        Column("cloud_coverage", Float),
        Column("description", String(100))
    )

    weather_clean = Table("weather_clean", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("city", String(50)),
        Column("datetime", DateTime),
        Column("temperature", Float),
        Column("humidity", Float),
        Column("pressure", Float),
        Column("wind_speed", Float)
    )

    weather_predictions = Table("weather_predictions", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("city", String(50)),
        Column("datetime", DateTime),
        Column("predicted_temp", Float),
        Column("predicted_humidity", Float),
        Column("model_name", String(100))
    )

    metadata.create_all(engine)
    print("Created tables: weather_raw, weather_clean, weather_predictions")