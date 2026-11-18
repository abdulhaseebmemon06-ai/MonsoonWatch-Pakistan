
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "pakistan_flood_locations_daily_weather_1981_2026.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("MONSOONWATCH PAKISTAN - DATA QUALITY ANALYSIS")
print("=" * 60)

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\n1. DATASET OVERVIEW")
print("-" * 40)

print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))
print("Columns:", df.columns.tolist())

print("Date Range:",
      df["date"].min(),
      "to",
      df["date"].max())

# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\n2. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()

missing_report = pd.DataFrame({
    "column": missing.index,
    "missing_values": missing.values,
    "missing_percentage": (
        missing.values / len(df) * 100
    ).round(3)
})

print(missing_report.to_string(index=False))

missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv",
    index=False
)

# ==========================================
# 4. DUPLICATE RECORDS
# ==========================================

print("\n3. DUPLICATE RECORDS")
print("-" * 40)

print("Full duplicate rows:", df.duplicated().sum())

print(
    "Duplicate location-date pairs:",
    df.duplicated(
        subset=["location", "date"]
    ).sum()
)

# ==========================================
# 5. LOCATION INFORMATION
# ==========================================

print("\n4. LOCATION ANALYSIS")
print("-" * 40)

print("Unique locations:", df["location"].nunique())

print("\nLocations by province:")

province_locations = (
    df.groupby("province")["location"]
    .nunique()
    .sort_values(ascending=False)
)

print(province_locations)

# ==========================================
# 6. MISSING DAILY RECORDS
# ==========================================

print("\n5. MISSING DAILY RECORDS")
print("-" * 40)

gap_results = []

for location, group in df.groupby("location"):

    dates = (
        group["date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(dates) == 0:
        continue

    expected_dates = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        freq="D"
    )

    missing_dates = expected_dates.difference(dates)

    gap_results.append({
        "location": location,
        "first_date": dates.min(),
        "last_date": dates.max(),
        "observed_days": len(dates),
        "missing_days": len(missing_dates)
    })

gap_report = pd.DataFrame(gap_results)

print(gap_report.to_string(index=False))

gap_report.to_csv(
    OUTPUT_DIR / "date_gaps_report.csv",
    index=False
)

# ==========================================
# 7. RAINFALL QUALITY
# ==========================================

print("\n6. RAINFALL ANALYSIS")
print("-" * 40)

rain = df["rainfall_mm"]

print("Minimum:", rain.min())
print("Maximum:", rain.max())
print("Mean:", round(rain.mean(), 3))
print("Median:", rain.median())
print("Standard Deviation:", round(rain.std(), 3))

print("Negative rainfall values:", (rain < 0).sum())

print("Zero rainfall days:", (rain == 0).sum())

print(
    "Heavy rainfall days (>=15 mm):",
    (rain >= 15).sum()
)

print(
    "Extreme rainfall days (>=50 mm):",
    (rain >= 50).sum()
)

# ==========================================
# 8. RAINFALL DISTRIBUTION
# ==========================================

print("\n7. RAINFALL DISTRIBUTION")
print("-" * 40)

rainfall_distribution = pd.DataFrame({
    "category": [
        "Zero",
        "Light (0-2.5 mm)",
        "Moderate (2.5-15 mm)",
        "Heavy (15-50 mm)",
        "Very Heavy (50+ mm)"
    ],
    "records": [
        (rain == 0).sum(),
        ((rain > 0) & (rain < 2.5)).sum(),
        ((rain >= 2.5) & (rain < 15)).sum(),
        ((rain >= 15) & (rain < 50)).sum(),
        (rain >= 50).sum()
    ]
})

rainfall_distribution["percentage"] = (
    rainfall_distribution["records"]
    / len(df) * 100
).round(2)

print(rainfall_distribution.to_string(index=False))

rainfall_distribution.to_csv(
    OUTPUT_DIR / "rainfall_distribution.csv",
    index=False
)

# ==========================================
# 9. EXTREME RAINFALL RECORDS
# ==========================================

print("\n8. TOP 20 HIGHEST RAINFALL RECORDS")
print("-" * 40)

extreme_records = (
    df.nlargest(20, "rainfall_mm")[
        ["date", "location", "province", "rainfall_mm"]
    ]
)

print(extreme_records.to_string(index=False))

extreme_records.to_csv(
    OUTPUT_DIR / "extreme_rainfall_records.csv",
    index=False
)

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n9. ANALYSIS COMPLETED")
print("-" * 40)

print("Reports saved in:", OUTPUT_DIR)

print("\nOriginal dataset has NOT been modified.")

print("=" * 60)
