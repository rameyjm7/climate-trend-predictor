#!/usr/bin/env python3
# ECE5984 – Weather EDA & cleaning (stocks-style)

import os
import pickle
import numpy as np
import pandas as pd

# ───────── Config ─────────
# Local pickle OR S3 path you downloaded locally (your batch_ingest writes to S3)
PICKLE_PATH = os.getenv("WEATHER_PKL", "weather_data.pkl")

# Reasonable bounds (tweak as needed)
TEMP_MIN, TEMP_MAX = -60.0, 60.0        # °C
HUM_MIN,  HUM_MAX  = 0.0,   100.0       # %
PRES_MIN, PRES_MAX = 870.0,  1085.0     # hPa
WIND_MIN, WIND_MAX = 0.0,    60.0       # m/s (~134 mph)

# ───────── Load raw data ─────────
with open(PICKLE_PATH, "rb") as f:
    raw_data = pickle.loads(f.read())

# Ensure DataFrame
if not isinstance(raw_data, pd.DataFrame):
    raise TypeError(f"Expected DataFrame in pickle, got {type(raw_data)}")

print("The Dataset looks like:")
print(raw_data)
print(getattr(raw_data, "shape", None))
print("====================================")

pd.set_option("display.max_columns", None)
print("Display first 5 rows")
print(raw_data.head().to_string())
print("====================================")

# ───────── Basic EDA ─────────
print("Basic Dataframe info")
print(raw_data.info())
print("====================================")

print("More detailed Dataframe info")
print(raw_data.describe(include="all").to_string())
print("====================================")

print("Number of Empty values in each column:")
print(raw_data.isnull().sum().sort_values(ascending=False))
print("====================================")

print("Number of Unique values in each column:")
print(raw_data.apply(pd.Series.nunique))
print("====================================")

print("Are there duplicate rows?")
print(raw_data.duplicated())
print("====================================")

# No multi-index gymnastics needed (weather schema is flat)
# Columns expected from your ingest: city, temp_celsius, humidity, pressure, wind_speed, timestamp

# Convert types and add datetime
df = raw_data.copy()
for col in ["temp_celsius", "humidity", "pressure", "wind_speed", "timestamp"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "timestamp" in df.columns:
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

# ───────── Split per “entity” (cities), like AAPL/AMZN/GOOGL ─────────
city_groups = {}
if "city" not in df.columns:
    raise KeyError("Expected 'city' column in weather data.")

for city in sorted(df["city"].dropna().unique()):
    city_groups[city] = df[df["city"] == city].copy()

# ───────── Clean each city df like the stocks example ─────────
def clean_city_df(g: pd.DataFrame) -> pd.DataFrame:
    # Drop NaNs in core columns
    core = ["temp_celsius", "humidity", "pressure", "wind_speed"]
    g = g.dropna(subset=[c for c in core if c in g.columns])

    # Remove outliers (analogous to >900 / <0.001 in stocks, but weather-realistic)
    if "temp_celsius" in g: g = g[(g["temp_celsius"] >= TEMP_MIN) & (g["temp_celsius"] <= TEMP_MAX)]
    if "humidity" in g:     g = g[(g["humidity"]      >= HUM_MIN)  & (g["humidity"]      <= HUM_MAX)]
    if "pressure" in g:     g = g[(g["pressure"]      >= PRES_MIN) & (g["pressure"]      <= PRES_MAX)]
    if "wind_speed" in g:   g = g[(g["wind_speed"]    >= WIND_MIN) & (g["wind_speed"]    <= WIND_MAX)]

    # Drop duplicates
    g = g.drop_duplicates()

    # Optional: collapse near-duplicates within the same minute, keep latest
    if "ts" in g:
        g["ts_min"] = g["ts"].dt.floor("min")
        g = g.sort_values("ts").drop_duplicates(subset=["ts_min"], keep="last")

    return g.reset_index(drop=True)

cleaned_groups = {city: clean_city_df(g) for city, g in city_groups.items()}

# Example: access like df_newyork, df_losangeles, etc., similar to df_aapl/df_amzn/df_googl
locals().update({f"df_{city.lower().replace(' ', '')}": g for city, g in cleaned_groups.items()})

# Print quick summaries
for city, g in cleaned_groups.items():
    print(f"─── {city} ───")
    print(g.describe().to_string())
    print(f"Rows after cleaning: {len(g)}")
    print("====================================")

# If you want a combined cleaned frame (like concatenating tickers later):
df_clean_all = pd.concat(cleaned_groups.values(), ignore_index=True).sort_values(["city", "ts"]).reset_index(drop=True)
print("Combined cleaned dataset shape:", df_clean_all.shape)
