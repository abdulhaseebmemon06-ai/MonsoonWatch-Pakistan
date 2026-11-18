from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "pakistan_flood_locations_daily_weather_1981_2026.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("\n=== DATASET OVERVIEW ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Locations: {df['location'].nunique()}")
print("Locations by province:")
print(df[["province", "location"]].drop_duplicates().groupby("province")["location"].count())
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())
print(f"\nDuplicate full rows: {df.duplicated().sum():,}")
print(f"Duplicate location-date pairs: {df.duplicated(['location', 'date']).sum():,}")
print("\nRainfall summary (mm):")
print(df["rainfall_mm"].describe(percentiles=[.5, .9, .95, .99]).round(3))
print(f"\nShare of days with zero rainfall: {(df['rainfall_mm'].eq(0).mean() * 100):.2f}%")

# Total monthly rainfall averaged across locations
monthly = df.assign(month_start=df["date"].dt.to_period("M").dt.to_timestamp()) \
            .groupby("month_start", as_index=False)["rainfall_mm"].mean()
plt.figure(figsize=(12, 5))
plt.plot(monthly["month_start"], monthly["rainfall_mm"])
plt.title("Average Daily Rainfall Across Selected Locations Over Time")
plt.xlabel("Date")
plt.ylabel("Average rainfall (mm/day)")
plt.tight_layout()
plt.savefig(OUT_DIR / "rainfall_trend.png", dpi=150)
plt.close()

# Seasonal pattern by calendar month
season = df.groupby("month", as_index=False)["rainfall_mm"].mean()
plt.figure(figsize=(9, 5))
plt.bar(season["month"].astype(str), season["rainfall_mm"])
plt.title("Average Daily Rainfall by Month")
plt.xlabel("Month")
plt.ylabel("Average rainfall (mm/day)")
plt.tight_layout()
plt.savefig(OUT_DIR / "rainfall_by_month.png", dpi=150)
plt.close()

# Average daily rainfall by location
loc = df.groupby("location", as_index=False)["rainfall_mm"].mean().sort_values("rainfall_mm")
plt.figure(figsize=(10, 7))
plt.barh(loc["location"], loc["rainfall_mm"])
plt.title("Average Daily Rainfall by Location")
plt.xlabel("Average rainfall (mm/day)")
plt.tight_layout()
plt.savefig(OUT_DIR / "rainfall_by_location.png", dpi=150)
plt.close()

print(f"\nCharts saved in: {OUT_DIR}")
