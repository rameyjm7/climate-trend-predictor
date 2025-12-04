#!/usr/bin/env python3
# Jacob M. Ramey
# ECE5984, Final Project
# Phase 6 – Feature Extraction / Sequence Preparation
# Loads cleaned weather JSON, scales features, builds time-series splits,
# writes X/y train/test to S3, and logs provenance to Amazon RDS.

import pandas as pd
import numpy as np
import pickle
import s3fs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from utils.provenance_utils import ProvenanceTimer   # <-- NEW (Phase 7)

# ---------------------------------------------------------------------------
# Main sequence preparation function
# ---------------------------------------------------------------------------
def prepare_sequences():

    CLEAN_DIR = "s3://ece5984-s3-rameyjm7/Project/transformed"
    SEQ_DIR   = "s3://ece5984-s3-rameyjm7/Project/sequences"

    with ProvenanceTimer(
        stage="feature_extraction_weather",
        input_source=f"{CLEAN_DIR}/combined_clean.json",
        output_target=SEQ_DIR
    ) as p:

        s3 = s3fs.S3FileSystem()

        # -------------------------------------------------------------------
        # Load cleaned JSON (from transform_phase)
        # -------------------------------------------------------------------
        try:
            with s3.open(f"{CLEAN_DIR}/combined_clean.json", "rb") as f:
                df = pd.read_json(f)
        except FileNotFoundError:
            raise RuntimeError("combined_clean.json not found — did transform run?")

        if df.empty:
            raise RuntimeError("Cleaned dataframe is empty — cannot build sequences.")

        # Sort by datetime for correct sliding-window behavior
        if "datetime" in df.columns:
            df = df.sort_values(by="datetime")
        else:
            df = df.sort_values(by="date")  # fallback

        # -------------------------------------------------------------------
        # Target variable
        # -------------------------------------------------------------------
        y = df["temperature"].astype(float).values

        # -------------------------------------------------------------------
        # LSTM input feature set (consistent with milestone)
        # -------------------------------------------------------------------
        feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

        # Select and coerce features
        X_raw = df[feature_cols].astype(float)

        # -------------------------------------------------------------------
        # Scale features (MinMaxScaler — EXACTLY like class)
        # -------------------------------------------------------------------
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)
        X_scaled = pd.DataFrame(X_scaled, index=df.index, columns=feature_cols)

        # Save scaler for inference
        with s3.open(f"{SEQ_DIR}/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # -------------------------------------------------------------------
        # TimeSeriesSplit (same pattern used in class — stocks HW3)
        # n_splits=10 → final split becomes train/test
        # -------------------------------------------------------------------
        tscv = TimeSeriesSplit(n_splits=10)

        X_train = X_test = y_train = y_test = None

        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled.iloc[train_idx].values
            X_test  = X_scaled.iloc[test_idx].values
            y_train = y[train_idx]
            y_test  = y[test_idx]

        # Safety check
        if X_train is None or X_test is None:
            raise RuntimeError("TimeSeriesSplit failed to generate train/test splits.")

        # -------------------------------------------------------------------
        # Save sequences to S3 (pickle format — matches class methodology)
        # -------------------------------------------------------------------
        with s3.open(f"{SEQ_DIR}/X_train_weather.pkl", "wb") as f:
            pickle.dump(X_train, f)
        with s3.open(f"{SEQ_DIR}/X_test_weather.pkl", "wb") as f:
            pickle.dump(X_test, f)
        with s3.open(f"{SEQ_DIR}/y_train_weather.pkl", "wb") as f:
            pickle.dump(y_train, f)
        with s3.open(f"{SEQ_DIR}/y_test_weather.pkl", "wb") as f:
            pickle.dump(y_test, f)

        # -------------------------------------------------------------------
        # Provenance Logging (Phase 7)
        # -------------------------------------------------------------------
        p.commit(
            status="SUCCESS",
            records_in=len(df),
            records_out=len(X_train) + len(X_test),
            extra={
                "features": feature_cols,
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "splits": 10
            }
        )

        print("Weather sequences prepared and stored to S3.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prepare_sequences()
