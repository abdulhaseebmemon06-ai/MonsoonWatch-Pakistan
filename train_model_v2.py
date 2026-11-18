
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
DATA_PATH = os.path.join(
    "data", "pakistan_flood_locations_daily_weather_1981_2026.csv"
)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "location", "rainfall_mm"])
df = df.sort_values(["location", "date"]).reset_index(drop=True)

grouped = df.groupby("location", sort=False)

# Target: rainfall on the following day
df["target_date"] = grouped["date"].shift(-1)
df["target_next_day_rainfall_mm"] = grouped["rainfall_mm"].shift(-1)

# Historical rainfall features: past observations only
df["rainfall_previous_day_mm"] = grouped["rainfall_mm"].shift(1)

df["rainfall_previous_3_days_mm"] = grouped["rainfall_mm"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=3).sum()
)

df["rainfall_previous_7_days_mm"] = grouped["rainfall_mm"].transform(
    lambda s: s.shift(1).rolling(7, min_periods=7).sum()
)

df["rainfall_previous_7_day_average_mm"] = grouped["rainfall_mm"].transform(
    lambda s: s.shift(1).rolling(7, min_periods=7).mean()
)

# Keep only consecutive daily records for the target
valid_next_day = (
    df["target_date"] == df["date"] + pd.Timedelta(days=1)
)
df = df.loc[valid_next_day].copy()

# Calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.day_of_year
df["day_of_week"] = df["date"].dt.day_of_week

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# --------------------------------------------------
# 3. Features
# --------------------------------------------------
numeric_features = [
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
    "rainfall_previous_day_mm",
    "rainfall_previous_3_days_mm",
    "rainfall_previous_7_days_mm",
    "rainfall_previous_7_day_average_mm",
]

categorical_features = ["location", "province"]
feature_columns = numeric_features + categorical_features
target_column = "target_next_day_rainfall_mm"

# The rolling features need enough preceding history.
df = df.dropna(subset=[
    "rainfall_previous_day_mm",
    "rainfall_previous_3_days_mm",
    "rainfall_previous_7_days_mm",
    "rainfall_previous_7_day_average_mm",
    target_column,
])

df = df.sort_values(["date", "location"]).reset_index(drop=True)

# --------------------------------------------------
# 4. Chronological 80/20 split
# --------------------------------------------------
unique_dates = np.sort(df["date"].unique())
split_index = int(len(unique_dates) * 0.80)
split_index = min(max(split_index, 1), len(unique_dates) - 1)
cutoff_date = unique_dates[split_index]

train = df[df["date"] < cutoff_date].copy()
test = df[df["date"] >= cutoff_date].copy()

X_train = train[feature_columns]
y_train = train[target_column]
X_test = test[feature_columns]
y_test = test[target_column]

print(f"Training rows: {len(train):,}")
print(f"Testing rows:  {len(test):,}")
print(f"Test period starts: {test['date'].min().date()}")

# --------------------------------------------------
# 5. Preprocessing and model
# --------------------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )),
])

# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------
print("\nTraining experimental Random Forest model...")
model.fit(X_train, y_train)

predictions = np.maximum(model.predict(X_test), 0)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = np.maximum(baseline.predict(X_test), 0)

def calculate_metrics(name, actual, predicted):
    return {
        "model": name,
        "samples": len(actual),
        "MAE_mm": mean_absolute_error(actual, predicted),
        "RMSE_mm": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

results = pd.DataFrame([
    calculate_metrics("Mean baseline", y_test, baseline_predictions),
    calculate_metrics("Random Forest V2", y_test, predictions),
])

print("\nEXPERIMENTAL MODEL RESULTS")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Heavy-rainfall evaluation
actual = y_test.to_numpy()
heavy_mask = actual >= 15

if heavy_mask.any():
    heavy_mae = mean_absolute_error(
        actual[heavy_mask], predictions[heavy_mask]
    )
    print("\nHEAVY-RAINFALL CHECK (actual >= 15 mm)")
    print(f"Heavy-rainfall test days: {heavy_mask.sum():,}")
    print(f"Heavy-day MAE: {heavy_mae:.3f} mm")
    print(f"Average actual: {actual[heavy_mask].mean():.3f} mm")
    print(f"Average predicted: {predictions[heavy_mask].mean():.3f} mm")
else:
    print("\nNo heavy-rainfall examples in the test set.")

# --------------------------------------------------
# 7. Save V2 outputs without overwriting V1
# --------------------------------------------------
model_path = os.path.join(OUTPUT_DIR, "best_model_v2.joblib")
metrics_path = os.path.join(OUTPUT_DIR, "evaluation_v2.csv")
predictions_path = os.path.join(OUTPUT_DIR, "predictions_v2.csv")
metadata_path = os.path.join(OUTPUT_DIR, "model_metadata_v2.json")

joblib.dump(model, model_path)
results.to_csv(metrics_path, index=False)

prediction_output = test[
    ["date", "target_date", "location", "province", target_column]
].copy()

prediction_output = prediction_output.rename(columns={
    "date": "feature_date",
    "target_date": "forecast_target_date",
    target_column: "actual_rainfall_mm",
})

prediction_output["predicted_rainfall_mm"] = predictions
prediction_output["absolute_error_mm"] = np.abs(
    prediction_output["actual_rainfall_mm"]
    - prediction_output["predicted_rainfall_mm"]
)

prediction_output.to_csv(predictions_path, index=False)

metadata = {
    "model": "RandomForestRegressor",
    "version": "V2 with historical rainfall features",
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": target_column,
    "test_start_date": str(test["date"].min().date()),
    "training_rows": int(len(train)),
    "testing_rows": int(len(test)),
}

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nSaved experimental model:", model_path)
print("Saved metrics:", metrics_path)
print("Saved predictions:", predictions_path)
print("Saved metadata:", metadata_path)
print("\nYour original best_model.joblib has not been changed.")
