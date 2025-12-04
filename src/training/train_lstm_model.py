#!/usr/bin/env python3
# Jacob M. Ramey
# ECE5984, Final Project
# Phase 6 – Model Training (TensorFlow LSTM)
# Loads sequence data from S3, trains an LSTM model, saves .keras model to S3,
# and logs provenance to Amazon RDS.

import numpy as np
import pickle
import s3fs
import tempfile
from keras.models import Sequential
from keras.layers import LSTM, Dense

from utils.provenance_utils import ProvenanceTimer   # <-- NEW (Phase 7)

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_lstm_weather():

    SEQ_DIR   = "s3://ece5984-s3-rameyjm7/Project/sequences"
    MODEL_DIR = "s3://ece5984-s3-rameyjm7/Project/models"

    with ProvenanceTimer(
        stage="train_lstm_weather",
        input_source=SEQ_DIR,
        output_target=f"{MODEL_DIR}/weather_lstm.keras"
    ) as p:

        s3 = s3fs.S3FileSystem()

        # -------------------------------------------------------------------
        # Load prepared sequences (X/y train/test)
        # -------------------------------------------------------------------
        try:
            X_train = np.load(s3.open(f"{SEQ_DIR}/X_train_weather.pkl"), allow_pickle=True)
            X_test  = np.load(s3.open(f"{SEQ_DIR}/X_test_weather.pkl"), allow_pickle=True)
            y_train = np.load(s3.open(f"{SEQ_DIR}/y_train_weather.pkl"), allow_pickle=True)
            y_test  = np.load(s3.open(f"{SEQ_DIR}/y_test_weather.pkl"), allow_pickle=True)
        except Exception as e:
            raise RuntimeError("Failed to load training sequences from S3: " + str(e))

        if X_train.size == 0 or X_test.size == 0:
            raise RuntimeError("Training or testing data is empty — cannot train model.")

        n_features = X_train.shape[1]

        # -------------------------------------------------------------------
        # Reshape for LSTM: (samples, timesteps=1, features)
        # -------------------------------------------------------------------
        X_train = X_train.reshape(X_train.shape[0], 1, n_features)
        X_test  = X_test.reshape(X_test.shape[0], 1, n_features)

        # -------------------------------------------------------------------
        # Build LSTM model (class architecture — HW3 pattern)
        # -------------------------------------------------------------------
        model = Sequential()
        model.add(LSTM(
            units=32,
            activation="relu",
            return_sequences=False,
            input_shape=(1, n_features)
        ))
        model.add(Dense(1))

        model.compile(
            loss="mean_squared_error",
            optimizer="adam"
        )

        # -------------------------------------------------------------------
        # Train model (25 epochs just like class)
        # -------------------------------------------------------------------
        history = model.fit(
            X_train, y_train,
            epochs=25,
            batch_size=8,
            verbose=1,
            shuffle=False,
            validation_data=(X_test, y_test)
        )

        # Get final metrics
        final_train_loss = float(history.history["loss"][-1])
        final_val_loss   = float(history.history["val_loss"][-1])

        # -------------------------------------------------------------------
        # Save model as .keras (required by you)
        # -------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_path = f"{temp_dir}/weather_lstm.keras"
            model.save(local_model_path)
            s3.put(local_model_path, f"{MODEL_DIR}/weather_lstm.keras")

        print("Weather LSTM model trained and uploaded to S3.")

        # -------------------------------------------------------------------
        # Provenance Log (Phase 7)
        # -------------------------------------------------------------------
        p.commit(
            status="SUCCESS",
            records_in=len(X_train),
            records_out=1,
            extra={
                "epochs": 25,
                "batch_size": 8,
                "features": n_features,
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "model_path": f"{MODEL_DIR}/weather_lstm.keras"
            }
        )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_lstm_weather()
