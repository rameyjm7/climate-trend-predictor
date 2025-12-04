#!/usr/bin/env python3
# Jacob M. Ramey
# ECE5984, Final Project
# Weather Transform Stage – converts latest raw weather pickle to clean JSON outputs
# Includes run-level logging for provenance and QA.

import pickle
import pandas as pd
import time
import json
from datetime import datetime
from s3fs.core import S3FileSystem

# ---------------------------------------------------------------------------
# Tunable sanity ranges (Celsius, %, hPa, m/s)
# ---------------------------------------------------------------------------
TEMP_MIN, TEMP_MAX = -60.0, 60.0
HUM_MIN, HUM_MAX = 0.0, 100.0
PRES_MIN, PRES_MAX = 870.0, 1085.0
WIND_MIN, WIND_MAX = 0.0, 60.0

REQUIRED_COLS = [
    "city",
    "date",
    "temperature",
    "temperature_min",
    "temperature_max",
    "humidity",
    "pressure",
    "wind_speed",
]

# ---------------------------------------------------------------------------
# Main transform function
# ---------------------------------------------------------------------------
def transform_weather_data(
    s3_dir_in="s3://ece5984-s3-rameyjm7/Project/batch_ingest",
    s3_dir_out="s3://ece5984-s3-rameyjm7/Project/transformed",
    s3_log_dir="s3://ece5984-s3-rameyjm7/Project/logs",
):
    """
    Load the most recent pickle from batch_ingest, clean it, and write curated JSON outputs.
    Produces:
      - combined_clean.json (all cities)
      - clean_<city>.json (per city)
      - stats_per_city.csv (summary stats for QA)
      - transformation_log.jsonl (run metadata)
    """
    start_time = time.time()
    s3 = S3FileSystem(anon=False)

    # -----------------------------------------------------------------------
    # Locate the latest pickle file automatically
    # -----------------------------------------------------------------------
    all_files = s3.ls(s3_dir_in)
    pkl_files = [f for f in all_files if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(f"No pickle files found in {s3_dir_in}")

    latest_pkl = sorted(pkl_files)[-1]
    print(f"Loading latest pickle: {latest_pkl}")

    # -----------------------------------------------------------------------
    # Load dataframe
    # -----------------------------------------------------------------------
    with s3.open(latest_pkl, "rb") as f:
        df = pickle.load(f)

    records_in = len(df)

    # -----------------------------------------------------------------------
    # Validate columns
    # -----------------------------------------------------------------------
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    df = df.copy()

    # -----------------------------------------------------------------------
    # Numeric type coercion
    # -----------------------------------------------------------------------
    for col in ["temperature", "temperature_min", "temperature_max",
                "humidity", "pressure", "wind_speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing any required field
    df = df.dropna(subset=REQUIRED_COLS)

    # -----------------------------------------------------------------------
    # Remove outliers
    # -----------------------------------------------------------------------
    mask_temp = (df["temperature"] >= TEMP_MIN) & (df["temperature"] <= TEMP_MAX)
    mask_hum = (df["humidity"] >= HUM_MIN) & (df["humidity"] <= HUM_MAX)
    mask_pres = (df["pressure"] >= PRES_MIN) & (df["pressure"] <= PRES_MAX)
    mask_wind = (df["wind_speed"] >= WIND_MIN) & (df["wind_speed"] <= WIND_MAX)
    df = df[mask_temp & mask_hum & mask_pres & mask_wind]

    # -----------------------------------------------------------------------
    # Normalize timestamp and deduplicate
    # -----------------------------------------------------------------------
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values(["city", "datetime"]).drop_duplicates()

    # -----------------------------------------------------------------------
    # Compute quick per-city stats
    # -----------------------------------------------------------------------
    stats = (
        df.groupby("city")[["temperature", "humidity", "pressure", "wind_speed"]]
        .agg(["count", "mean", "std", "min", "max"])
    )

    # -----------------------------------------------------------------------
    # Write curated outputs (data warehouse layer)
    # -----------------------------------------------------------------------
    combined_path = f"{s3_dir_out}/combined_clean.json"
    with s3.open(combined_path, "w") as f:
        df.to_json(f, orient="records", date_format="iso")
    print("Wrote combined JSON:", combined_path)

    for city, g in df.groupby("city"):
        safe_city = city.lower().replace(" ", "_")
        city_path = f"{s3_dir_out}/clean_{safe_city}.json"
        with s3.open(city_path, "w") as f:
            g.to_json(f, orient="records", date_format="iso")
        print("Wrote:", city_path)

    stats_path = f"{s3_dir_out}/stats_per_city.csv"
    with s3.open(stats_path, "w") as f:
        stats.to_csv(f)
    print("Wrote stats CSV:", stats_path)

    # -----------------------------------------------------------------------
    # Transformation logging (metadata for provenance)
    # -----------------------------------------------------------------------
    records_out = len(df)
    duration = round(time.time() - start_time, 2)

    log_entry = {
        "run_id": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": latest_pkl,
        "records_in": records_in,
        "records_out": records_out,
        "processing_time_s": duration,
        "status": "success",
        "notes": f"Cleaned and exported {records_out} records from {records_in} input rows."
    }

    # Use absolute S3 log path
    log_path = f"{s3_log_dir}/transformation_log.jsonl"
    with s3.open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print("Transformation log updated:", log_path)
    print("Transformation summary:", json.dumps(log_entry, indent=2))

    return df

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    transform_weather_data()
