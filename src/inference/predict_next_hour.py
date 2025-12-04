#!/usr/bin/env python3
# Jacob M. Ramey
# ECE5984, Final Project
# Phase 6 – Model Inference
# Uses last cleaned weather record from RDS, runs LSTM inference,
# inserts prediction into RDS, and logs provenance (Phase 7).

import numpy as np
import pickle
import pandas as pd
import s3fs
import tempfile
from sqlalchemy import create_engine
from keras.models import load_model

from utils.provenance_utils import ProvenanceTimer   # <-- NEW

# ---------------------------------------------------------------------------
# Main Prediction Function
# ---------------------------------------------------------------------------
def predict_next_hour():

    SEQ_DIR   = "s3://ece5984-s3-rameyjm7/Project/sequences"
    MODEL_DIR = "s3://ece5984-s3-rameyjm7/Project/models"

    # Update these with your actual RDS credentials
    RDS_USER = "YOURUSER"
    RDS_PASS = "YOURPASS"
    RDS_HOST = "YOUR-RDS-ENDPOINT.amazonaws.com"
    RDS_DB   = "weatherdb"

    with ProvenanceTimer(
        stage="predict_next_hour_weather",
        input_source="RDS: weather_clean (latest 1 row)",
        output_target="RDS: weather_predictions"
    ) as p:

        s3 = s3fs.S3FileSystem()

        # -------------------------------------------------------------------
        # Load scaler from S3
        # -------------------------------------------------------------------
        try:
            scaler = pickle.load(s3.open(f"{SEQ_DIR}/scaler.pkl", "rb"))
        except Exception as e:
            raise RuntimeError("Failed to load scaler.pkl from S3: " + str(e))

        # -------------------------------------------------------------------
        # Load trained .keras model from S3 -> /tmp
        # -------------------------------------------------------------------
        try:
            with s3.open(f"{MODEL_DIR}/weather_lstm.keras", "rb") as f:
                with open("/tmp/weather_lstm.keras", "wb") as tmp_f:
                    tmp_f.write(f.read())
            model = load_model("/tmp/weather_lstm.keras")
        except Exception as e:
            raise RuntimeError("Failed to load LSTM model from S3: " + str(e))

        # -------------------------------------------------------------------
        # Connect to Amazon RDS
        # -------------------------------------------------------------------
        try:
            engine = create_engine(
                f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}/{RDS_DB}"
            )
        except Exception as e:
            raise RuntimeError("Failed to connect to Amazon RDS: " + str(e))

        # -------------------------------------------------------------------
        # Get latest weather record from RDS
        # -------------------------------------------------------------------
        query = """
            SELECT *
            FROM weather_clean
            ORDER BY datetime DESC
            LIMIT 1;
        """

        df = pd.read_sql(query, engine)

        if df.empty:
            raise RuntimeError("No rows available in RDS weather_clean table.")

        df = df.sort_values("datetime")

        # -------------------------------------------------------------------
        # Prepare LSTM input row
        # -------------------------------------------------------------------
        feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

        try:
            X_input = df[feature_cols].astype(float).values[-1:]
        except KeyError as e:
            raise RuntimeError("RDS weather_clean missing required feature columns: " + str(e))

        # Scale + reshape
        X_scaled = scaler.transform(X_input)
        X_scaled = X_scaled.reshape(1, 1, X_scaled.shape[1])

        # -------------------------------------------------------------------
        # Run LSTM prediction
        # -------------------------------------------------------------------
        pred = float(model.predict(X_scaled)[0][0])

        # -------------------------------------------------------------------
        # Insert prediction into RDS predictions table
        # -------------------------------------------------------------------
        insert_sql = """
            INSERT INTO weather_predictions (datetime, prediction)
            VALUES (NOW(), %s);
        """

        with engine.begin() as conn:
            conn.execute(insert_sql, (pred,))

        print(f"Prediction stored in RDS: {pred:.4f}")

        # -------------------------------------------------------------------
        # Provenance Logging (Phase 7)
        # -------------------------------------------------------------------
        p.commit(
            status="SUCCESS",
            records_in=1,
            records_out=1,
            extra={
                "last_weather_row": df.to_dict(orient="records")[0],
                "prediction": pred,
                "model_path": f"{MODEL_DIR}/weather_lstm.keras"
            }
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    predict_next_hour()
