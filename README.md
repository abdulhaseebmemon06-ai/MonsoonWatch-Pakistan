# MonsoonWatch Pakistan

**Project:** Pakistan Monsoon Rainfall Forecasting and Heavy-Rainfall Alert System Using Machine Learning

This starter project uses historical daily weather observations for selected locations in Pakistan. It is intended for academic learning and demonstration, not operational weather warnings.

## Project structure

- `data/` — daily weather CSV and its data dictionary
- `eda.py` — initial data-quality checks and exploratory charts
- `train_model.py` — next-day rainfall model training and evaluation
- `app.py` — interactive Streamlit dashboard for dataset exploration
- `outputs/` — generated charts and model outputs

## 1. Set up on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can instead run the commands using `\.venv\Scripts\python.exe` directly.

## 2. Run the initial EDA

```powershell
python eda.py
```

The script prints dataset size, date range, locations, missing values, duplicates, and rainfall summary. It saves charts in `outputs/`.

## 3. Launch the dashboard

```powershell
streamlit run app.py
```

## 4. Train the first forecasting models

```powershell
python train_model.py
```

The model uses weather observations from day *t* to predict rainfall at the same location on day *t+1*. A chronological date split is used to reduce future-data leakage. Model quality must be checked before making any claims.

## Important limitation

This dataset contains weather measurements, not verified flood-event labels. The project forecasts rainfall and can flag heavy rainfall according to a chosen threshold; it does **not** establish or predict actual floods. Weather observations are based on selected location coordinates and should not be interpreted as measurements for every point in a district.
