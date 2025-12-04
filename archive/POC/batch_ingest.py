# Jacob M. Ramey
# ECE5984, Final Project
# Weather Data Ingestion for Multiple Cities using OpenWeatherMap Free API
# Saves Pickle files to data lake (batch_ingest) for later transformation.

import os
import requests
import pandas as pd
import pickle
import s3fs
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Missing API_KEY environment variable")

CITIES = ["New York", "Los Angeles", "Chicago", "Miami", "Seattle"]
S3_PATH = "s3://ece5984-s3-rameyjm7/Project/batch_ingest"
LOCAL_FILE = "weather_forecast_all_cities.pkl"
URL = "http://api.openweathermap.org/data/2.5/forecast"  # Free 5-day/3-hour endpoint

# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
def fetch_forecast(city):
    """Fetch 5-day forecast data for a single city."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(URL, params=params)
    if response.status_code != 200:
        raise RuntimeError("Error fetching forecast for " + city + ": " + response.text)
    return response.json()

# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------
def ingest_weather_forecast(cities):
    all_records = []

    for city in cities:
        try:
            print(f"Fetching forecast for {city}...")
            data = fetch_forecast(city)
            forecasts = data.get("list", [])
            for f in forecasts:
                record = {
                    "city": city,
                    "date": datetime.utcfromtimestamp(f["dt"]).strftime("%m/%d/%Y %H:%M"),
                    "temperature": f["main"].get("temp"),
                    "temperature_min": f["main"].get("temp_min"),
                    "temperature_max": f["main"].get("temp_max"),
                    "humidity": f["main"].get("humidity"),
                    "pressure": f["main"].get("pressure"),
                    "wind_speed": f["wind"].get("speed"),
                    "cloud_coverage": f["clouds"].get("all"),
                    "precipitation": f.get("rain", {}).get("3h", 0.0)
                                     + f.get("snow", {}).get("3h", 0.0),
                    "description": f["weather"][0]["description"]
                                   if f.get("weather") else None,
                }
                all_records.append(record)
            print(f"  Retrieved {len(forecasts)} forecast entries.")
        except Exception as e:
            print(f"  Skipping {city}: {e}")

    if not all_records:
        print("No data retrieved.")
        return None

    # -----------------------------------------------------------------------
    # Save DataFrame as Pickle (data lake layer)
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_records)
    df.sort_values(by=["city", "date"], inplace=True)

    # Save locally
    with open(LOCAL_FILE, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved local pickle file: {LOCAL_FILE}")

    # Upload to S3 (data lake)
    try:
        s3 = s3fs.S3FileSystem()
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        s3_path = f"{S3_PATH}/weather_forecast_all_cities_{timestamp}.pkl"
        with s3.open(s3_path, "wb") as f:
            pickle.dump(df, f)
        print(f"Uploaded pickle to S3: {s3_path}")
    except Exception as e:
        print("S3 upload skipped or failed:", str(e))

    return df

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingest_weather_forecast(CITIES)
