import numpy as np
import pickle
import s3fs
from sqlalchemy import create_engine
from keras.models import load_model

def predict_next_hour():

    s3 = s3fs.S3FileSystem()

    SEQ_DIR = "s3://ece5984-s3-rameyjm7/Project/sequences"
    MODEL_DIR = "s3://ece5984-s3-rameyjm7/Project/models"

    # Load scaler
    scaler = pickle.load(s3.open(f"{SEQ_DIR}/scaler.pkl", "rb"))

    # Load trained model
    with s3.open(f"{MODEL_DIR}/weather_lstm.keras", "rb") as f:
        with open("/tmp/tmp_model.keras", "wb") as local_f:
            local_f.write(f.read())

    model = load_model("/tmp/tmp_model.keras")

    # Connect to RDS (same style as your HW4)
    engine = create_engine(
        "mysql+pymysql://USERNAME:PASSWORD@RDS-ENDPOINT/weatherdb"
    )

    # Grab the last N rows needed to form input
    df = pd.read_sql("SELECT * FROM weather_clean ORDER BY date DESC LIMIT 1", engine)
    df = df.sort_values("date")

    feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]
    X_input = df[feature_cols].astype(float).values[-1:]

    # Scale
    X_scaled = scaler.transform(X_input)
    X_scaled = X_scaled.reshape(1, 1, X_scaled.shape[1])

    pred = float(model.predict(X_scaled)[0][0])

    # Insert into predictions table
    with engine.begin() as conn:
        conn.execute(
            "INSERT INTO weather_predictions(date, prediction) VALUES (NOW(), %s)", 
            (pred,)
        )

    print("Weather prediction stored in RDS.")
