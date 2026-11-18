
from pathlib import Path
import json
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data" /
    "pakistan_flood_locations_daily_weather_1981_2026.csv"
)

MODEL_PATH = ROOT / "outputs" / "best_model_v2.joblib"
METRICS_PATH = ROOT / "outputs" / "evaluation_v2.csv"
METADATA_PATH = ROOT / "outputs" / "model_metadata_v2.json"


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="MonsoonWatch Pakistan",
    page_icon="🌧️",
    layout="wide"
)

st.title("MonsoonWatch Pakistan")
st.subheader("Pakistan Monsoon Rainfall Analysis and Forecasting")

st.caption(
    "Academic prototype using historical weather data. "
    "Not an official weather-warning service."
)


# ---------------------------------------------------------
# CHECK REQUIRED FILES
# ---------------------------------------------------------

required_files = [
    DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    METADATA_PATH
]

missing_files = [
    path.name for path in required_files if not path.exists()
]

if missing_files:
    st.error(
        "Required project files are missing:\n\n"
        + "\n".join(missing_files)
    )
    st.stop()


# ---------------------------------------------------------
# LOAD DATA AND MODEL
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


df = load_data()
model = load_model()
metadata = load_metadata()
metrics = load_metrics()

# Use the exact feature order from V2 metadata.
expected_features = metadata["feature_columns"]


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tabs = st.tabs([
    "Historical Analysis",
    "Predict Rainfall",
    "Model Evaluation"
])


# =========================================================
# TAB 1: HISTORICAL ANALYSIS
# =========================================================

with tabs[0]:

    st.header("Explore Historical Weather Data")

    st.sidebar.header("Historical Data Filters")

    provinces = ["All"] + sorted(
        df["province"].dropna().unique().tolist()
    )

    selected_province = st.sidebar.selectbox(
        "Province",
        provinces
    )

    filtered = df.copy()

    if selected_province != "All":
        filtered = filtered[
            filtered["province"] == selected_province
        ]

    locations = ["All"] + sorted(
        filtered["location"].dropna().unique().tolist()
    )

    selected_location = st.sidebar.selectbox(
        "Location",
        locations
    )

    if selected_location != "All":
        filtered = filtered[
            filtered["location"] == selected_location
        ]

    if filtered.empty:
        st.warning("No records match the selected filters.")

    else:

        min_date = filtered["date"].min().date()
        max_date = filtered["date"].max().date()

        date_range = st.sidebar.date_input(
            "Historical date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

            start = pd.Timestamp(date_range[0])
            end = pd.Timestamp(date_range[1])

            filtered = filtered[
                filtered["date"].between(start, end)
            ]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Weather records",
            f"{len(filtered):,}"
        )

        c2.metric(
            "Locations shown",
            f"{filtered['location'].nunique():,}"
        )

        c3.metric(
            "Average rainfall",
            f"{filtered['rainfall_mm'].mean():.2f} mm/day"
            if len(filtered) else "N/A"
        )

        c4.metric(
            "Maximum daily rainfall",
            f"{filtered['rainfall_mm'].max():.2f} mm"
            if len(filtered) else "N/A"
        )

        if not filtered.empty:

            st.markdown("### Rainfall over time")

            daily = filtered.groupby("date")["rainfall_mm"].mean()

            st.line_chart(
                daily,
                y_label="Rainfall (mm/day)"
            )

            left, right = st.columns(2)

            with left:

                st.markdown("### Average rainfall by month")

                month_data = (
                    filtered.groupby("month")["rainfall_mm"].mean()
                )

                st.bar_chart(
                    month_data,
                    y_label="Rainfall (mm/day)"
                )

            with right:

                st.markdown("### Average rainfall by location")

                location_data = (
                    filtered.groupby("location")["rainfall_mm"]
                    .mean()
                    .sort_values()
                )

                st.bar_chart(
                    location_data,
                    y_label="Rainfall (mm/day)"
                )

            st.markdown("### Preview records")

            st.dataframe(
                filtered.head(100),
                use_container_width=True
            )

        else:
            st.warning("No records match the selected date range.")


# =========================================================
# TAB 2: RAINFALL PREDICTION
# =========================================================

with tabs[1]:

    st.header("Next-Day Rainfall Prediction")

    st.write(
        "Enter weather observations for the day immediately before "
        "your forecast date. The V2 Random Forest model estimates "
        "rainfall in millimetres for the selected location."
    )

    st.warning(
        "This is an academic machine-learning estimate, not an "
        "official weather forecast. It does not predict floods "
        "and does not fetch live weather data."
    )

    province_options = sorted(
        df["province"].dropna().unique().tolist()
    )

    province_for_prediction = st.selectbox(
        "Forecast province",
        province_options,
        key="prediction_province"
    )

    province_locations = sorted(
        df.loc[
            df["province"] == province_for_prediction,
            "location"
        ].dropna().unique().tolist()
    )

    location_for_prediction = st.selectbox(
        "Forecast location",
        province_locations,
        key="prediction_location"
    )

    location_history = (
        df[df["location"] == location_for_prediction]
        .sort_values("date")
        .reset_index(drop=True)
    )

    latest = location_history.iloc[-1]

    latest_date = latest["date"].date()

    st.caption(
        f"Latest dataset record for this location: {latest_date}. "
        "Default values are historical examples, not live readings."
    )

    # -----------------------------------------------------
    # DEFAULT HISTORICAL RAINFALL FEATURES
    # -----------------------------------------------------

    # These defaults use records before the latest dataset row.
    # They are examples only; users should enter actual values
    # for the relevant observation date.

    previous_history = location_history.iloc[:-1]

    previous_rainfall = previous_history[
        "rainfall_mm"
    ].tail(7).tolist()

    previous_day_default = (
        float(previous_rainfall[-1])
        if len(previous_rainfall) >= 1 else 0.0
    )

    previous_3_days_default = (
        float(sum(previous_rainfall[-3:]))
        if len(previous_rainfall) >= 3 else 0.0
    )

    previous_7_days_default = (
        float(sum(previous_rainfall))
        if len(previous_rainfall) >= 7 else 0.0
    )

    previous_7_day_average_default = (
        float(np.mean(previous_rainfall))
        if len(previous_rainfall) >= 7 else 0.0
    )

    # -----------------------------------------------------
    # FORECAST DATE
    # -----------------------------------------------------

    default_forecast_date = max(
        date.today() + timedelta(days=1),
        latest_date + timedelta(days=1)
    )

    forecast_date = st.date_input(
        "Date to estimate rainfall for",
        value=default_forecast_date,
        min_value=date.today() + timedelta(days=1),
        key="forecast_date"
    )

    observation_date = forecast_date - timedelta(days=1)

    st.write(
        f"Weather observation date expected: **{observation_date}**"
    )

    # -----------------------------------------------------
    # PREDICTION FORM
    # -----------------------------------------------------

    with st.form("rainfall_prediction_form"):

        st.markdown("#### Weather observations")

        st.caption(
            "Enter weather readings for the observation date. "
            "The rainfall-history fields refer to days before "
            "that observation date."
        )

        a, b, c = st.columns(3)

        with a:

            rainfall = st.number_input(
                "Rainfall on observation date (mm)",
                min_value=0.0,
                max_value=1000.0,
                value=float(latest["rainfall_mm"]),
                step=0.1
            )

            temperature = st.number_input(
                "Average temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=float(latest["temperature_c"]),
                step=0.1
            )

            max_temperature = st.number_input(
                "Maximum temperature (°C)",
                min_value=-20.0,
                max_value=65.0,
                value=float(latest["max_temperature_c"]),
                step=0.1
            )

        with b:

            min_temperature = st.number_input(
                "Minimum temperature (°C)",
                min_value=-30.0,
                max_value=50.0,
                value=float(latest["min_temperature_c"]),
                step=0.1
            )

            humidity = st.number_input(
                "Relative humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(latest["relative_humidity_percent"]),
                step=1.0
            )

            wind = st.number_input(
                "Wind speed (m/s)",
                min_value=0.0,
                max_value=100.0,
                value=float(latest["wind_speed_mps"]),
                step=0.1
            )

        with c:

            previous_day_rainfall = st.number_input(
                "Rainfall on previous day (mm)",
                min_value=0.0,
                max_value=1000.0,
                value=previous_day_default,
                step=0.1
            )

            previous_3_days = st.number_input(
                "Total rainfall in previous 3 days (mm)",
                min_value=0.0,
                max_value=3000.0,
                value=previous_3_days_default,
                step=0.1
            )

            previous_7_days = st.number_input(
                "Total rainfall in previous 7 days (mm)",
                min_value=0.0,
                max_value=7000.0,
                value=previous_7_days_default,
                step=0.1
            )

            previous_7_day_average = st.number_input(
                "Average rainfall in previous 7 days (mm)",
                min_value=0.0,
                max_value=1000.0,
                value=previous_7_day_average_default,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Estimate Next-Day Rainfall"
        )

    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    if submitted:

        observation = pd.Timestamp(observation_date)

        input_row = pd.DataFrame([{

            "rainfall_mm": rainfall,

            "temperature_c": temperature,

            "max_temperature_c": max_temperature,

            "min_temperature_c": min_temperature,

            "relative_humidity_percent": humidity,

            "wind_speed_mps": wind,

            "latitude": float(latest["latitude"]),

            "longitude": float(latest["longitude"]),

            "year": observation.year,

            "month": observation.month,

            "day": observation.day,

            "day_of_year": observation.dayofyear,

            "day_of_week": observation.dayofweek,

            "month_sin": np.sin(
                2 * np.pi * observation.month / 12
            ),

            "month_cos": np.cos(
                2 * np.pi * observation.month / 12
            ),

            "rainfall_previous_day_mm": previous_day_rainfall,

            "rainfall_previous_3_days_mm": previous_3_days,

            "rainfall_previous_7_days_mm": previous_7_days,

            "rainfall_previous_7_day_average_mm":
                previous_7_day_average,

            "location": location_for_prediction,

            "province": province_for_prediction

        }])

        # Ensure the input columns match V2 metadata exactly.
        input_row = input_row[expected_features]

        prediction = max(
            0.0,
            float(model.predict(input_row)[0])
        )

        st.markdown("### Estimated rainfall")

        st.metric(
            label=f"Rainfall estimate for {forecast_date}",
            value=f"{prediction:.2f} mm"
        )

        if prediction < 2.5:

            st.write(
                "Model output: very little or no rainfall estimated."
            )

        elif prediction < 15:

            st.write(
                "Model output: light to moderate rainfall range."
            )

        elif prediction < 50:

            st.write(
                "Model output: substantial rainfall estimate."
            )

        else:

            st.write(
                "Model output: high rainfall estimate. "
                "Verify with official forecasts."
            )

        st.caption(
            "These display ranges are illustrative, not validated "
            "warning thresholds. The model may be inaccurate, "
            "particularly for rare heavy-rainfall events. "
            "Prediction quality also depends on the accuracy of "
            "the weather observations entered."
        )


# =========================================================
# TAB 3: MODEL EVALUATION
# =========================================================

with tabs[2]:

    st.header("Model Evaluation")

    st.write(
        f"Selected model: **{metadata['model']}**"
    )

    st.write(
        f"Model version: **{metadata['version']}**"
    )

    st.write(
        f"Chronological test period starts: "
        f"**{metadata['test_start_date']}**"
    )

    st.write(
        f"Training samples: **{metadata['training_rows']:,}**"
    )

    st.write(
        f"Testing samples: **{metadata['testing_rows']:,}**"
    )

    st.markdown("### Comparison on the held-out test period")

    st.dataframe(
        metrics.round(4),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # V2 PERFORMANCE METRICS
    # -----------------------------------------------------

    v2_result = metrics[
        metrics["model"] == "Random Forest V2"
    ]

    if not v2_result.empty:

        result = v2_result.iloc[0]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Mean Absolute Error",
            f"{result['MAE_mm']:.4f} mm"
        )

        m2.metric(
            "Root Mean Squared Error",
            f"{result['RMSE_mm']:.4f} mm"
        )

        m3.metric(
            "R² Score",
            f"{result['R2']:.4f}"
        )

    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    st.markdown("### What do the metrics mean?")

    st.write(
        "- **MAE (mm):** Average absolute difference between "
        "estimated and observed rainfall. Lower is better."
    )

    st.write(
        "- **RMSE (mm):** Penalizes larger prediction errors "
        "more heavily. Lower is better."
    )

    st.write(
        "- **R²:** Measures how much variation is explained "
        "by the model compared with a mean-based reference. "
        "A low or negative value indicates weak performance."
    )

    st.info(
        "Rainfall data contain many zero-rainfall days and "
        "relatively rare heavy-rainfall events. The model "
        "performs better on ordinary rainfall patterns than "
        "on extreme rainfall events. Good overall MAE alone "
        "does not prove that heavy rainfall can be predicted "
        "reliably. This project has no verified flood-event labels."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "MonsoonWatch Pakistan | Student Data Science Project | "
    "Historical dataset and Random Forest V2 prototype"
)
