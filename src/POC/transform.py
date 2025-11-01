#!/usr/bin/env python3
# ECE5984 – Weather transform, stocks-style
import pickle
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem

# ---- Tunable sanity ranges (Celsius, %, hPa, m/s) ----
TEMP_MIN, TEMP_MAX = -60.0, 60.0          # plausible for major cities
HUM_MIN,  HUM_MAX  = 0.0,   100.0         # relative humidity
PRES_MIN, PRES_MAX = 870.0,  1085.0       # sea-level pressure extremes
WIND_MIN, WIND_MAX = 0.0,    60.0         # 60 m/s ~ 134 mph

REQUIRED_COLS = ["city", "temp_celsius", "humidity", "pressure", "wind_speed", "timestamp"]

def transform_weather_data(
    s3_dir_in: str = "s3://ece5984-s3-rameyjm7/HW2/batch_ingest",
    s3_dir_out: str = "s3://ece5984-s3-rameyjm7/HW2/transformed",
    input_filename: str = "weather_data.pkl",
):
    """
    Read raw weather pickle from S3, clean like the stock transform, and write cleaned data back to S3.
    Produces:
      - combined_clean.pkl (all cities)
      - one clean_<city>.pkl per city
      - stats_per_city.csv (summary stats for quick QA)
    """
    s3 = S3FileSystem(anon=False)

    # ---------- Load ----------
    with s3.open(f"{s3_dir_in}/{input_filename}", "rb") as f:
        df = pickle.loads(f.read())

    # Ensure expected columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    # ---------- Type coercion & basic cleanup ----------
    df = df.copy()
    for col in ["temp_celsius", "humidity", "pressure", "wind_speed", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing any core field
    df = df.dropna(subset=REQUIRED_COLS)

    # ---------- Remove outliers / impossible values ----------
    mask_temp = (df["temp_celsius"] >= TEMP_MIN) & (df["temp_celsius"] <= TEMP_MAX)
    mask_hum  = (df["humidity"]      >= HUM_MIN)  & (df["humidity"]      <= HUM_MAX)
    mask_pres = (df["pressure"]      >= PRES_MIN) & (df["pressure"]      <= PRES_MAX)
    mask_wind = (df["wind_speed"]    >= WIND_MIN) & (df["wind_speed"]    <= WIND_MAX)

    df = df[mask_temp & mask_hum & mask_pres & mask_wind]

    # ---------- Timestamp normalization & de-dup ----------
    # Convert epoch -> pandas datetime (UTC), then floor to minute to collapse near-duplicates
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["ts_min"] = df["ts"].dt.floor("min")

    # Keep the *latest* sample within each (city, minute) bucket
    df = df.sort_values(["city", "ts"]).drop_duplicates(subset=["city", "ts_min"], keep="last")

    # Drop any exact duplicate rows as final pass
    df = df.drop_duplicates()

    # ---------- Sort for stable downstream reads ----------
    df = df.sort_values(["city", "ts"]).reset_index(drop=True)

    # ---------- (Optional) quick stats for QA ----------
    stats = (
        df.groupby("city")[["temp_celsius", "humidity", "pressure", "wind_speed"]]
        .agg(["count", "mean", "std", "min", "max"])
    )

    # ---------- Write outputs ----------
    # Combined
    with s3.open(f"{s3_dir_out}/combined_clean.pkl", "wb") as f:
        f.write(pickle.dumps(df))

    # Per-city parquet & pickle (pickle to mirror the stocks example)
    for city, g in df.groupby("city"):
        safe_city = city.lower().replace(" ", "_")
        with s3.open(f"{s3_dir_out}/clean_{safe_city}.pkl", "wb") as f:
            f.write(pickle.dumps(g))

    # Stats CSV for a fast eyeball check
    with s3.open(f"{s3_dir_out}/stats_per_city.csv", "w") as f:
        stats.to_csv(f)

    return df  # convenient for tests/notebooks


if __name__ == "__main__":
    transform_weather_data()