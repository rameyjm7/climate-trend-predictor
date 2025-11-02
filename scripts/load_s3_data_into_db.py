#!/usr/bin/env python3
# Load cleaned JSON from S3 into local SQLite database

import pandas as pd
from sqlalchemy import create_engine
from db_utils import get_engine

def load_clean_data_to_db():
    s3_path = "s3://ece5984-s3-rameyjm7/Project/transformed/combined_clean.json"
    print(f"Loading data from {s3_path} ...")
    df = pd.read_json(s3_path)

    print(f"Loaded {len(df)} records from S3")
    if df.empty:
        print("WARNING: No data found at this path.")
        return

    # Connect to local SQLite DB
    engine = get_engine("local")

    # Write to table
    df.to_sql("weather_clean", engine, if_exists="replace", index=False)
    print("Data successfully written to weather_clean")

if __name__ == "__main__":
    load_clean_data_to_db()