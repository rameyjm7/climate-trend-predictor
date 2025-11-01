# Jacob M. Ramey
# ECE5984, Homework 2
# this produces a pickle file of the data we ingest
import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import requests
import time

API_KEY = ""  # I took out the API key for security reasons before submitting to canvas
CITIES = ["New York", "Los Angeles", "Chicago", "Miami", "Seattle"]
URL = "http://api.openweathermap.org/data/2.5/weather"


def ingest_weather_data():
    weather_rows = []

    for city in CITIES:
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(URL, params=params)

        if response.status_code == 200:
            data = response.json()
            row = {
                "city": city,
                "temp_celsius": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "timestamp": time.time(),
            }
            weather_rows.append(row)
        else:
            print(f"Error fetching {city}: {response.text}")

    # Convert to DataFrame
    df = pd.DataFrame(weather_rows)

    # Add noise like the stock example
    for col in ["temp_celsius", "humidity", "pressure", "wind_speed"]:
        if col in df.columns:
            df.loc[df.sample(frac=0.1).index, col] = np.nan
            df.loc[df.sample(frac=0.005).index, col] = 1000
            df.loc[df.sample(frac=0.005).index, col] = 0

    # Duplicate some rows
    df = pd.concat([df, df.sample(frac=0.1)])

    # Save to S3
    s3 = S3FileSystem()
    DIR = "s3://ece5984-s3-rameyjm7/HW2/batch_ingest"
    with s3.open(f"{DIR}/weather_data.pkl", "wb") as f:
        f.write(pickle.dumps(df))

    print("Weather data ingestion complete. Saved to S3:", f"{DIR}/weather_data.pkl")