# Jacob M. Ramey
# ECE5984, Final Project
# Weather Data Ingestion for Multiple Cities using OpenWeatherMap API
# Saves Pickle files to S3 (batch_ingest layer) + Logs provenance to Amazon RDS.

import os
import requests
import pandas as pd
import pickle
import s3fs
from datetime import datetime

from utils.provenance_utils import ProvenanceTimer   # <-- NEW (Phase 7)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Missing API_KEY environment variable.")

S3_PATH = "s3://ece5984-s3-rameyjm7/Project/batch_ingest"
LOCAL_FILE = "weather_forecast_all_cities.pkl"
URL = "http://api.openweathermap.org/data/2.5/forecast"  # Free 5-day/3-hour endpoint

# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
def fetch_forecast(city):
    """Fetch 5-day forecast JSON data for a single city."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(URL, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"{city} API error: {response.text}")

    return response.json()

# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------
def ingest_weather_forecast(cities=None):
    """
    Ingest 5-day weather forecasts and save combined dataset locally + to S3.
    Includes provenance logging to Amazon RDS.
    """

    # Default cities (can be overridden by Airflow/Kafka)
    if cities is None:
        cities = ["New York", "Los Angeles", "Chicago", "Miami", "Seattle"]

    # -----------------------------------------------------------------------
    # Provenance Logging (START)
    # -----------------------------------------------------------------------
    with ProvenanceTimer(
        stage="ingest_weather_forecast",
        input_source="OpenWeatherMap: 5-day forecast API",
        output_target=S3_PATH
    ) as p:

        all_records = []

        # -------------------------------------------------------------------
        # API calls for each city
        # -------------------------------------------------------------------
        for city in cities:
            try:
                print(f"Fetching forecast for {city}...")
                data = fetch_forecast(city)

                forecasts = data.get("list", [])
                for entry in forecasts:
                    record = {
                        "city": city,
                        "date": datetime.utcfromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "temperature": entry["main"].get("temp"),
                        "temperature_min": entry["main"].get("temp_min"),
                        "temperature_max": entry["main"].get("temp_max"),
                        "humidity": entry["main"].get("humidity"),
                        "pressure": entry["main"].get("pressure"),
                        "wind_speed": entry["wind"].get("speed"),
                        "cloud_coverage": entry["clouds"].get("all"),
                        "precipitation": entry.get("rain", {}).get("3h", 0.0)
                                          + entry.get("snow", {}).get("3h", 0.0),
                        "description": entry["weather"][0]["description"] if entry.get("weather") else None,
                    }
                    all_records.append(record)

                print(f"  Retrieved {len(forecasts)} forecast entries for {city}.")

            except Exception as e:
                print(f"  Skipping {city}: {e}")

        # -------------------------------------------------------------------
        # No data case
        # -------------------------------------------------------------------
        if not all_records:
            print("No forecast data retrieved. Nothing to upload.")
            p.commit(status="FAILURE", records_in=0, records_out=0)
            return None

        # -------------------------------------------------------------------
        # Save dataframe locally
        # -------------------------------------------------------------------
        df = pd.DataFrame(all_records)
        df.sort_values(by=["city", "date"], inplace=True)

        with open(LOCAL_FILE, "wb") as f:
            pickle.dump(df, f)

        print(f"Saved local pickle file: {LOCAL_FILE}")

        # -------------------------------------------------------------------
        # Save to S3
        # -------------------------------------------------------------------
        try:
            s3 = s3fs.S3FileSystem()
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            s3_file_path = f"{S3_PATH}/weather_forecast_all_cities_{timestamp}.pkl"

            with s3.open(s3_file_path, "wb") as f:
                pickle.dump(df, f)

            print(f"Uploaded pickle to S3: {s3_file_path}")

            # -------------------------------------------------------------------
            # Commit provenance (SUCCESS)
            # -------------------------------------------------------------------
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
    ingest_weather_forecast()
