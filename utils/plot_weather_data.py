#!/usr/bin/env python3

"""
Weather Plot Utilities
Jacob M. Ramey
ECE5984 Final Project

This module provides visualization helpers for:
1. Historical weather data
2. Live weather data
3. Forecast data (model predictions)
4. Combined actual vs predicted overlay plots
"""

import os
import matplotlib
matplotlib.use("Agg")  # safe for headless environments (Airflow, servers)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.config import LOCAL_TRANSFORMED, LOCAL_LIVE, LOCAL_MODELS, USE_AWS

if USE_AWS:
    import s3fs
    from src.config import S3_LIVE, S3_MODELS


# ----------------------------------------------------------------------
# Load latest live weather file
# ----------------------------------------------------------------------
def load_latest_live():
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        files = s3.glob(f"{S3_LIVE}/weather_live_all_cities_*.pkl")
        if not files:
            return None
        newest = sorted(files)[-1]
        with s3.open(newest, "rb") as f:
            return pd.read_pickle(f)
    else:
        local_files = [
            os.path.join(LOCAL_LIVE, f)
            for f in os.listdir(LOCAL_LIVE)
            if f.startswith("weather_live_all_cities_")
        ]
        if not local_files:
            return None
        newest = sorted(local_files)[-1]
        return pd.read_pickle(newest)


# ----------------------------------------------------------------------
# Plot historical data for a city
# ----------------------------------------------------------------------
def plot_historical_weather(city, output_dir=None):

    df = pd.read_json(os.path.join(LOCAL_TRANSFORMED, "combined_clean.json"))
    df = df[df["city"] == city].sort_values("datetime")

    if df.empty:
        print(f"No historical data available for city: {city}")
        return None

    if output_dir is None:
        output_dir = os.path.join(LOCAL_MODELS, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df["datetime"], df["temperature"], label="Temperature (C)")
    plt.plot(df["datetime"], df["humidity"], label="Humidity")
    plt.plot(df["datetime"], df["wind_speed"], label="Wind Speed")
    plt.plot(df["datetime"], df["precipitation"], label="Precipitation")

    plt.title(f"{city} Historical Weather Trends")
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    out = os.path.join(output_dir, f"{city}_historical_weather.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    return out


# ----------------------------------------------------------------------
# Plot live vs historical comparison
# ----------------------------------------------------------------------
def plot_live_vs_historical(city, hours=48, output_dir=None):

    df_hist = pd.read_json(os.path.join(LOCAL_TRANSFORMED, "combined_clean.json"))
    df_hist = df_hist[df_hist["city"] == city].sort_values("datetime").tail(hours)

    df_live = load_latest_live()
    if df_live is not None:
        df_live_city = df_live[df_live["city"] == city]
    else:
        df_live_city = None

    if output_dir is None:
        output_dir = os.path.join(LOCAL_MODELS, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(df_hist["datetime"], df_hist["temperature"], label="Historical Temp")

    if df_live_city is not None and not df_live_city.empty:
        plt.scatter(df_live_city["datetime"], df_live_city["temperature"],
                    color="red", label="Live Temp", zorder=5)

    plt.title(f"{city} Live vs Historical Temperature")
    plt.xlabel("Datetime")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    out = os.path.join(output_dir, f"{city}_live_vs_historical.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    return out


# ----------------------------------------------------------------------
# Plot model forecast (timestamps, pred_c list)
# ----------------------------------------------------------------------
def plot_forecast(city, timestamps, preds_c, output_dir=None):

    if output_dir is None:
        output_dir = os.path.join(LOCAL_MODELS, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, preds_c, marker="o", label="Predicted Temp")

    plt.title(f"{city} Model Forecast")
    plt.xlabel("Datetime")
    plt.ylabel("Predicted Temperature (C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    out = os.path.join(output_dir, f"{city}_forecast_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    return out


# ----------------------------------------------------------------------
# Combined: Historical + Forecast overlay
# ----------------------------------------------------------------------
def plot_historical_and_forecast(city, timestamps, preds_c, hours_back=48, output_dir=None):

    df_hist = pd.read_json(os.path.join(LOCAL_TRANSFORMED, "combined_clean.json"))
    df_hist = df_hist[df_hist["city"] == city].sort_values("datetime").tail(hours_back)

    if output_dir is None:
        output_dir = os.path.join(LOCAL_MODELS, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(df_hist["datetime"], df_hist["temperature"], label="Historical Temp")
    plt.plot(timestamps, preds_c, marker="o", label="Forecast Temp", linewidth=2)

    plt.title(f"{city} Historical + Forecast Temperature")
    plt.xlabel("Datetime")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    out = os.path.join(output_dir, f"{city}_historical_forecast_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    return out

# ----------------------------------------------------------------------
# Run plotting utilities directly
# ----------------------------------------------------------------------
if __name__ == "__main__":

    city = "Seattle"
    print(f"Running weather plot utilities for city: {city}")

    # Historical plot
    out1 = plot_historical_weather(city)
    print(f"Historical plot saved: {out1}")

    # Live vs historical plot
    out2 = plot_live_vs_historical(city)
    print(f"Live vs historical plot saved: {out2}")

    # Optional: Run a small inference to plot forecast
    try:
        from src.inference.predict_weather import predict_next_n_hours

        timestamps, preds_c, preds_f = predict_next_n_hours(
            city=city,
            steps=8
        )

        # Forecast-only plot
        out3 = plot_forecast(city, timestamps, preds_c)
        print(f"Forecast plot saved: {out3}")

        # Historical + Forecast overlay
        out4 = plot_historical_and_forecast(city, timestamps, preds_c)
        print(f"Historical + forecast overlay saved: {out4}")

    except Exception as e:
        print("Forecast plotting skipped due to error:")
        print(str(e))
