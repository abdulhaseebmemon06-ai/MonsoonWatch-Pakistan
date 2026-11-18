
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs" / "best_model_v2.joblib"
METADATA_PATH = BASE_DIR / "outputs" / "model_metadata_v2.json"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "pakistan_flood_locations_daily_weather_1981_2026.csv"
)


# --------------------------------------------------
# 2. LOAD MODEL AND METADATA
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURE_COLUMNS = metadata["feature_columns"]


# --------------------------------------------------
# 3. LOAD LOCATION INFORMATION
# --------------------------------------------------

location_data = pd.read_csv(
    DATA_PATH,
    usecols=["location", "province", "latitude", "longitude"]
)

location_data = location_data.drop_duplicates(
    subset=["location"]
)

LOCATION_INFO = (
    location_data
    .set_index("location")
    .to_dict(orient="index")
)


def get_provinces():
    """Return available provinces."""
    return sorted(location_data["province"].unique().tolist())


def get_locations(province):
    """Return locations belonging to a province."""
    locations = location_data[
        location_data["province"] == province
    ]["location"]

    return sorted(locations.unique().tolist())


# --------------------------------------------------
# 4. RAINFALL CATEGORY AND INTERPRETATION
# --------------------------------------------------

def get_rainfall_category(prediction):
    """
    Categorize predicted rainfall using illustrative
    project thresholds, not official warning levels.
    """

    if prediction < 2.5:
        return {
            "category": "Very Little / No Rain",
            "interpretation": (
                "Very little or no rainfall is predicted "
                "for the following day."
            ),
            "severity": "low"
        }

    elif prediction < 15:
        return {
            "category": "Light to Moderate",
            "interpretation": (
                "Light to moderate rainfall is predicted. "
                "Some rainfall may occur in the selected location."
            ),
            "severity": "moderate"
        }

    elif prediction < 50:
        return {
            "category": "Substantial",
            "interpretation": (
                "Substantial rainfall is predicted. "
                "Monitor official weather updates."
            ),
            "severity": "high"
        }

    else:
        return {
            "category": "High Rainfall",
            "interpretation": (
                "High rainfall is predicted. "
                "Check official forecasts and local advisories."
            ),
            "severity": "very-high"
        }


# --------------------------------------------------
# 5. RAINFALL PREDICTION FUNCTION
# --------------------------------------------------

def predict_next_day_rainfall(inputs):
    """
    Predict next-day rainfall using the saved V2 model.

    Historical rainfall inputs must represent values
    from dates before the observation date.
    """

    location = inputs["location"]
    province = inputs["province"]

    if location not in LOCATION_INFO:
        raise ValueError("Invalid location selected.")

    location_details = LOCATION_INFO[location]

    if location_details["province"] != province:
        raise ValueError("Location does not belong to this province.")

    # Observation date
    observation_date = pd.to_datetime(
        inputs["observation_date"]
    )

    # Weather and rainfall inputs
    rainfall = float(inputs["rainfall_mm"])
    temperature = float(inputs["temperature_c"])
    max_temperature = float(inputs["max_temperature_c"])
    min_temperature = float(inputs["min_temperature_c"])
    humidity = float(inputs["relative_humidity_percent"])
    wind_speed = float(inputs["wind_speed_mps"])

    previous_day = float(inputs["rainfall_previous_day_mm"])
    previous_3_days = float(inputs["rainfall_previous_3_days_mm"])
    previous_7_days = float(inputs["rainfall_previous_7_days_mm"])
    previous_7_day_average = float(
        inputs["rainfall_previous_7_day_average_mm"]
    )

    # --------------------------------------------------
    # 6. BASIC INPUT VALIDATION
    # --------------------------------------------------

    rainfall_values = [
        rainfall,
        previous_day,
        previous_3_days,
        previous_7_days,
        previous_7_day_average
    ]

    if any(value < 0 for value in rainfall_values):
        raise ValueError("Rainfall values cannot be negative.")

    if not 0 <= humidity <= 100:
        raise ValueError("Humidity must be between 0 and 100.")

    if wind_speed < 0:
        raise ValueError("Wind speed cannot be negative.")

    if min_temperature > max_temperature:
        raise ValueError(
            "Minimum temperature cannot exceed maximum temperature."
        )

    if not min_temperature <= temperature <= max_temperature:
        raise ValueError(
            "Temperature must be between minimum and maximum temperature."
        )

    # --------------------------------------------------
    # 7. FEATURE ENGINEERING
    # Same calculations used during V2 training
    # --------------------------------------------------

    month = observation_date.month

    features = {
        "rainfall_mm": rainfall,
        "temperature_c": temperature,
        "max_temperature_c": max_temperature,
        "min_temperature_c": min_temperature,
        "relative_humidity_percent": humidity,
        "wind_speed_mps": wind_speed,
        "latitude": float(location_details["latitude"]),
        "longitude": float(location_details["longitude"]),
        "year": observation_date.year,
        "month": month,
        "day": observation_date.day,
        "day_of_year": observation_date.dayofyear,
        "day_of_week": observation_date.dayofweek,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "rainfall_previous_day_mm": previous_day,
        "rainfall_previous_3_days_mm": previous_3_days,
        "rainfall_previous_7_days_mm": previous_7_days,
        "rainfall_previous_7_day_average_mm": previous_7_day_average,
        "location": location,
        "province": province
    }

    # Ensure exact feature order from model metadata
    input_data = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    # --------------------------------------------------
    # 8. MODEL PREDICTION
    # --------------------------------------------------

    prediction = float(model.predict(input_data)[0])

    # Rainfall cannot be negative
    prediction = max(0.0, prediction)

    # Following day
    target_date = observation_date + pd.Timedelta(days=1)

    # Get rainfall category and interpretation
    rainfall_info = get_rainfall_category(prediction)

    # --------------------------------------------------
    # 9. RETURN COMPLETE PREDICTION RESULT
    # --------------------------------------------------

    return {
        "predicted_rainfall_mm": round(prediction, 2),

        "rainfall_category": rainfall_info["category"],

        "interpretation": rainfall_info["interpretation"],

        "severity": rainfall_info["severity"],

        "observation_date": observation_date.strftime("%Y-%m-%d"),

        "prediction_date": target_date.strftime("%Y-%m-%d"),

        "location": location,

        "province": province,

        "temperature_c": temperature,

        "max_temperature_c": max_temperature,

        "min_temperature_c": min_temperature,

        "humidity": humidity,

        "wind_speed_mps": wind_speed,

        "current_rainfall_mm": rainfall,

        "previous_day_rainfall_mm": previous_day,

        "previous_3_days_mm": previous_3_days,

        "previous_7_days_mm": previous_7_days,

        "previous_7_day_average_mm": previous_7_day_average,

        "model_version": metadata["version"],

        "notice": (
            "Academic prediction only. Rainfall categories are "
            "illustrative and are not official weather warnings."
        )
    }
