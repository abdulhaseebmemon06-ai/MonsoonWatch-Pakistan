
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "pakistan_flood_locations_daily_weather_1981_2026.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

dates_to_check = ["2020-05-29", "2022-08-06", "2022-07-31"]

columns = [
    "date",
    "location",
    "province",
    "latitude",
    "longitude",
    "rainfall_mm",
    "temperature_c",
    "max_temperature_c",
    "min_temperature_c",
    "relative_humidity_percent",
    "wind_speed_mps"
]

for date in dates_to_check:
    print(f"\n{'=' * 65}")
    print(f"RECORDS FOR {date}")
    print("=" * 65)

    result = df[df["date"] == pd.Timestamp(date)]
    result = result.sort_values("rainfall_mm", ascending=False)

    print(result[columns].head(10).to_string(index=False))
