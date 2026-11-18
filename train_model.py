
from pathlib import Path
import json

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

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "pakistan_flood_locations_daily_weather_1981_2026.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["location", "date"]).copy()

# Predict rainfall for the next calendar day at the same location.
df["target_date"] = df.groupby("location")["date"].shift(-1)
df["target_next_day_rainfall_mm"] = (
    df.groupby("location")["rainfall_mm"].shift(-1)
)

is_next_calendar_day = (
    df["target_date"] - df["date"] == pd.Timedelta(days=1)
)
df = df[is_next_calendar_day].copy()

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
target = "target_next_day_rainfall_mm"

df = df.dropna(subset=[target]).copy()

# Chronological split: the test set contains later dates.
dates = np.sort(df["date"].unique())
if len(dates) < 10:
    raise ValueError("Not enough unique dates to train and evaluate the model.")

cutoff = pd.Timestamp(dates[int(len(dates) * 0.80)])
train = df[df["date"] < cutoff].copy()
test = df[df["date"] >= cutoff].copy()

numeric_features = [c for c in features if c not in ("location", "province")]
categorical_features = ["location", "province"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

baseline = DummyRegressor(strategy="mean")
baseline.fit(train[features], train[target])
baseline_predictions = np.maximum(
    baseline.predict(test[features]), 0
)

model.fit(train[features], train[target])
predictions = np.maximum(model.predict(test[features]), 0)

def calculate_metrics(name, actual, predicted):
    return {
        "model": name,
        "MAE_mm": mean_absolute_error(actual, predicted),
        "RMSE_mm": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }

results = [
    calculate_metrics("Mean baseline", test[target], baseline_predictions),
    calculate_metrics("Random Forest", test[target], predictions),
]

results_df = pd.DataFrame(results).sort_values("MAE_mm")
results_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

# Save the fitted preprocessing and prediction pipeline.
joblib.dump(model, OUT_DIR / "best_model.joblib")

metadata = {
    "best_model": "Random Forest",
    "target": target,
    "date_cutoff": str(cutoff.date()),
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target_definition": "Rainfall at the same location on the next calendar day",
    "note": (
        "Academic prototype. Evaluated on a chronological holdout. "
        "Rainfall forecast only; no verified flood labels."
    ),
}

with open(OUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nModel evaluation:")
print(results_df.to_string(index=False))
print("\nSaved model: outputs/best_model.joblib")
print("Saved evaluation: outputs/model_comparison.csv")
print("Saved metadata: outputs/model_metadata.json")
print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")
print(f"Test period starts: {cutoff.date()}")
