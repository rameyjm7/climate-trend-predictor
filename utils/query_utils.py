from db_utils import get_engine
import pandas as pd

engine = get_engine("local")
query = """
SELECT city, AVG(temperature) AS avg_temp, AVG(humidity) AS avg_humidity
FROM weather_clean
GROUP BY city
ORDER BY avg_temp DESC;
"""
df = pd.read_sql(query, engine)
print(df)