#!/usr/bin/env python3

import pandas as pd
import numpy as np
import s3fs
from sqlalchemy import create_engine
import os

# Load env variables (same as inference)
RDS_USER = os.getenv("RDS_USER")
RDS_PASS = "0xUPw3#ooQr><D]o!iD!m4L~BJbJ" #os.getenv("RDS_PASS")
RDS_HOST = os.getenv("RDS_HOST")
DB_NAME  = "rameyjm7"   # your only DB

if not all([RDS_USER, RDS_PASS, RDS_HOST]):
    raise RuntimeError("Missing RDS_USER / RDS_PASS / RDS_HOST env vars")

def load_clean_weather_into_rds():

    print("Loading cleaned weather data into Amazon RDS...")

    s3 = s3fs.S3FileSystem()
    CLEAN_DIR = "s3://ece5984-s3-rameyjm7/Project/transformed"

    # Load cleaned JSON (created by transform_weather_data.py)
    with s3.open(f"{CLEAN_DIR}/combined_clean.json", "rb") as f:
        df = pd.read_json(f)

    # Connect WITHOUT selecting database first (HW4 pattern)
    engine_root = create_engine(f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}")

    # Ensure database exists (HW4 pattern)
    with engine_root.connect() as conn:
        exists = conn.execute(f"SHOW DATABASES LIKE '{DB_NAME}'").fetchone()
        if not exists:
            print(f"Creating database '{DB_NAME}'")
            conn.execute(f"CREATE DATABASE {DB_NAME}")

    # Reconnect to your personal database
    engine = create_engine(f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}/{DB_NAME}")

    # AUTO-CREATE weather_clean table
    df.to_sql("weather_clean", con=engine, if_exists="replace", index=False)

    print("weather_clean successfully loaded into RDS.")
    print(f"Row count: {len(df)}")


if __name__ == "__main__":
    load_clean_weather_into_rds()
