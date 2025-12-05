#!/usr/bin/env python3
import os
import time
import pickle
import requests
import pandas as pd
import s3fs
from datetime import datetime, timedelta

from src.config import (
    USE_AWS, API_KEY,
    CITIES,
    LOCAL_HISTORICAL,
    S3_BATCH
)

URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_historical(city):
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    g = requests.get(geocode_url, params={"name": city}).json()
    lat = g["results"][0]["latitude"]
    lon = g["results"][0]["longitude"]

    start = "2020-01-01"
    end   = datetime.utcnow().strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation,windspeed_10m"
    }

    r = requests.get(URL, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"Failed for {city}: {r.text}")

    data = r.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "city": city,
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relativehumidity_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed": hourly["windspeed_10m"]
    })

    return df


def ingest_weather_historical():
    print("Downloading historical weather...")

    all_df = []

    for city in CITIES:
        print(f"Downloading historical weather for {city}...")
        df = fetch_historical(city)
        print(f"  Retrieved {len(df)} rows for {city}.")
        all_df.append(df)
        time.sleep(1.2)

    final = pd.concat(all_df, ignore_index=True)

    local_path = os.path.join(LOCAL_HISTORICAL, "historical_weather_all_cities.csv")
    final.to_csv(local_path, index=False)
    print(f"Saved local historical dataset: {local_path}")

    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        s3_path = f"{S3_BATCH}/historical_weather_all_cities_{ts}.csv"
        with s3.open(s3_path, "wb") as f:
            final.to_csv(f, index=False)
        print(f"Uploaded to S3: {s3_path}")


if __name__ == "__main__":
    ingest_weather_historical()
