pip install -r requirements.txt
source env.txt
python3 src/ingestion/ingest_weather_forecast.py
python3 src/ingestion/ingest_weather_historical.py
python3 src/ingestion/ingest_weather_live.py
