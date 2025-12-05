#!/usr/bin/env python3
# Jacob M. Ramey
# Live Weather Data Ingestion using OpenWeatherMap Current Weather API
# Saves Pickle files to S3 (live_ingest layer) + Logs provenance to Amazon RDS.

import os
import requests
import pandas as pd
import pickle
import s3fs
from datetime import datetime

from utils.provenance_utils import ProvenanceTimer   # Phase 7

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Missing API_KEY environment variable.")

S3_PATH = "s3://ece5984-s3-rameyjm7/Project/live_ingest"
LOCAL_FILE = "weather_live_all_cities.pkl"
URL = "https://api.openweathermap.org/data/2.5/weather"   # Live endpoint

# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
def fetch_live_weather(city):
    """Fetch current weather JSON data for a single city."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    r = requests.get(URL, params=params)

    if r.status_code != 200:
        raise RuntimeError(f"{city} LIVE API error: {r.text}")

    j = r.json()

    return {
        "city": city,
        "date": datetime.utcfromtimestamp(j["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": j["main"].get("temp"),
        "temperature_min": j["main"].get("temp_min"),
        "temperature_max": j["main"].get("temp_max"),
        "humidity": j["main"].get("humidity"),
        "pressure": j["main"].get("pressure"),
        "wind_speed": j["wind"].get("speed"),
        "cloud_coverage": j["clouds"].get("all"),
        "precipitation": j.get("rain", {}).get("1h", 0.0)
                          + j.get("snow", {}).get("1h", 0.0),
        "description": j["weather"][0]["description"]
                       if j.get("weather") else None,
    }

# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------
def ingest_weather_live(cities=None):
    """
    Ingest real-time weather conditions for each city
    and save combined dataset locally + to S3.
    Includes provenance logging to Amazon RDS.
    """

    if cities is None:
        cities = ["New York", "Los Angeles", "Chicago", "Miami", "Seattle"]

    with ProvenanceTimer(
        stage="ingest_weather_live",
        input_source="OpenWeatherMap: Live Weather API",
        output_target=S3_PATH
    ) as p:

        all_records = []

        # ---------------------------------------------------------------
        # API calls for each city
        # ---------------------------------------------------------------
        for city in cities:
            try:
                print(f"Fetching LIVE weather for {city}...")
                row = fetch_live_weather(city)
                all_records.append(row)
                print("  Retrieved 1 live observation.")
            except Exception as e:
                print(f"  Skipping {city}: {e}")

        if not all_records:
            print("No LIVE data retrieved. Nothing to upload.")
            p.commit(status="FAILURE", records_in=0, records_out=0)
            return None

        # ---------------------------------------------------------------
        # Save dataframe locally
        # ---------------------------------------------------------------
        df = pd.DataFrame(all_records)
        df.sort_values(by=["city", "date"], inplace=True)

        with open(LOCAL_FILE, "wb") as f:
            pickle.dump(df, f)

        print(f"Saved local pickle file: {LOCAL_FILE}")

        # ---------------------------------------------------------------
        # Upload to S3
        # ---------------------------------------------------------------
        try:
            s3 = s3fs.S3FileSystem()
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            s3_file_path = f"{S3_PATH}/weather_live_all_cities_{timestamp}.pkl"

            with s3.open(s3_file_path, "wb") as f:
                pickle.dump(df, f)

            print(f"Uploaded LIVE weather pickle to S3: {s3_file_path}")

            p.commit(
                status="SUCCESS",
                records_in=len(cities),
                records_out=len(df),
                extra={"s3_path": s3_file_path, "cities": cities}
            )

        except Exception as e:
            print("S3 upload skipped or failed:", str(e))
            p.commit(
                status="FAILURE",
                records_in=len(cities),
                records_out=len(df),
                extra={"error": str(e)}
            )

        return df

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingest_weather_live()
