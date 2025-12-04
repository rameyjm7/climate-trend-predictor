import numpy as np
import pickle
import s3fs
import tempfile
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_lstm_weather():

    s3 = s3fs.S3FileSystem()

    SEQ_DIR = "s3://ece5984-s3-rameyjm7/Project/sequences"
    MODEL_DIR = "s3://ece5984-s3-rameyjm7/Project/models"

    # Load prepared sequences
    X_train = np.load(s3.open(f"{SEQ_DIR}/X_train_weather.pkl"), allow_pickle=True)
    X_test  = np.load(s3.open(f"{SEQ_DIR}/X_test_weather.pkl"), allow_pickle=True)
    y_train = np.load(s3.open(f"{SEQ_DIR}/y_train_weather.pkl"), allow_pickle=True)
    y_test  = np.load(s3.open(f"{SEQ_DIR}/y_test_weather.pkl"), allow_pickle=True)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Build LSTM model (exact class architecture)
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=8,
        verbose=1,
        shuffle=False,
        validation_data=(X_test, y_test)
    )

    # Save as .keras instead of .h5
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = f"{temp_dir}/weather_lstm.keras"
        model.save(save_path)
        s3.put(save_path, f"{MODEL_DIR}/weather_lstm.keras")

    print("Weather LSTM model trained and uploaded to S3.")
