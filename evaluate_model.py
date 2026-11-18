
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "pakistan_flood_locations_daily_weather_1981_2026.csv"
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# 1. Load the dataset and saved model
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
model = joblib.load(MODEL_PATH)

# 2. Recreate the next-calendar-day rainfall target
df = df.sort_values(["location", "date"]).copy()

df["target_date"] = df.groupby("location")["date"].shift(-1)
df["actual_rainfall_mm"] = (
    df.groupby("location")["rainfall_mm"].shift(-1)
)

next_day = (
    df["target_date"] - df["date"]
    == pd.Timedelta(days=1)
)
df = df[next_day].copy()
df = df.dropna(subset=["actual_rainfall_mm"]).copy()

# 3. Recreate the same features used during training
df["day_of_year"] = df["date"].dt.dayofyear
df["day_of_week"] = df["date"].dt.dayofweek
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

features = [
    "rainfall_mm",
    "temperature_c",
    "max_temperature_c",
    "min_temperature_c",
    "relative_humidity_percent",
    "wind_speed_mps",
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "day_of_year",
    "day_of_week",
    "month_sin",
    "month_cos",
    "location",
    "province",
]

# 4. Recreate the chronological train/test split
dates = np.sort(df["date"].unique())
cutoff = pd.Timestamp(dates[int(len(dates) * 0.80)])

train = df[df["date"] < cutoff].copy()
test = df[df["date"] >= cutoff].copy()

# 5. Predict using the saved model
X_train = train[features]
X_test = test[features]

y_train = train["actual_rainfall_mm"]
y_test = test["actual_rainfall_mm"]

model_predictions = np.maximum(model.predict(X_test), 0)

# Baseline: predict the average rainfall from training data
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = np.maximum(
    baseline.predict(X_test), 0
)

# 6. Calculate evaluation metrics
def metrics(name, actual, predicted):
    return {
        "model": name,
        "samples": len(actual),
        "MAE_mm": mean_absolute_error(actual, predicted),
        "RMSE_mm": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }

overall = pd.DataFrame([
    metrics("Mean baseline", y_test, baseline_predictions),
    metrics("Saved model", y_test, model_predictions),
])

print("\nOVERALL TEST RESULTS")
print(overall.to_string(index=False))

# 7. Evaluate performance by actual rainfall amount
predictions_df = test[
    ["date", "target_date", "location", "province"]
].copy()

predictions_df["actual_rainfall_mm"] = y_test.to_numpy()
predictions_df["predicted_rainfall_mm"] = model_predictions
predictions_df["absolute_error_mm"] = np.abs(
    predictions_df["actual_rainfall_mm"]
    - predictions_df["predicted_rainfall_mm"]
)

# These are analysis groups, not official warning thresholds.
bins = [-0.001, 0.1, 2.5, 15, np.inf]
labels = [
    "Near-zero (<0.1 mm)",
    "Light (0.1 to <2.5 mm)",
    "Moderate (2.5 to <15 mm)",
    "Heavy (15 mm or more)",
]

predictions_df["actual_rainfall_group"] = pd.cut(
    predictions_df["actual_rainfall_mm"],
    bins=bins,
    labels=labels,
    right=False,
)

group_results = (
    predictions_df.groupby(
        "actual_rainfall_group",
        observed=False,
    )
    .agg(
        samples=("actual_rainfall_mm", "size"),
        MAE_mm=("absolute_error_mm", "mean"),
        mean_actual_mm=("actual_rainfall_mm", "mean"),
        mean_predicted_mm=("predicted_rainfall_mm", "mean"),
    )
    .reset_index()
)

print("\nERRORS BY ACTUAL RAINFALL GROUP")
print(group_results.to_string(index=False))

# 8. Check heavy-rainfall days separately
heavy = predictions_df[
    predictions_df["actual_rainfall_mm"] >= 15
]

print("\nHEAVY-RAINFALL CHECK (actual >= 15 mm)")
print(f"Heavy-rainfall test days: {len(heavy):,}")

if len(heavy) > 0:
    print(
        "Heavy-day MAE:",
        round(heavy["absolute_error_mm"].mean(), 3),
        "mm",
    )
    print(
        "Average actual rainfall:",
        round(heavy["actual_rainfall_mm"].mean(), 3),
        "mm",
    )
    print(
        "Average predicted rainfall on those days:",
        round(heavy["predicted_rainfall_mm"].mean(), 3),
        "mm",
    )
else:
    print("No heavy-rainfall days found in this test period.")

# 9. Save detailed results
overall.to_csv(OUT_DIR / "evaluation_overall.csv", index=False)
group_results.to_csv(
    OUT_DIR / "evaluation_by_rainfall_group.csv",
    index=False,
)
predictions_df.to_csv(
    OUT_DIR / "evaluation_predictions.csv",
    index=False,
)

print("\nEvaluation complete.")
print("Test period starts:", cutoff.date())
print("Overall results: outputs/evaluation_overall.csv")
print("Rainfall groups: outputs/evaluation_by_rainfall_group.csv")
print("Predictions: outputs/evaluation_predictions.csv")
