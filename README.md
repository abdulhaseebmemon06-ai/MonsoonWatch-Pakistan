# MonsoonWatch Pakistan

**Pakistan Monsoon Rainfall Analysis and Next-Day Forecasting Using Machine Learning**

MonsoonWatch Pakistan is a web-based machine learning application designed to analyze historical rainfall patterns and predict next-day rainfall for selected locations across Pakistan.

## Project Features

- User Registration and Login
- Interactive Dashboard
- Next-Day Rainfall Prediction
- Location-Based Weather Analysis
- Historical Rainfall Analysis
- Machine Learning Model Evaluation
- Project Information

## Dataset

The project uses historical daily weather observations containing:

- 365,574 weather records
- 22 locations across 4 provinces
- Historical data from January 1981 to June 2026

## Machine Learning Model

The project uses a Random Forest Regressor to predict next-day rainfall.

**Model Evaluation (Version 2):**
- MAE: 1.8871 mm
- RMSE: 8.6384 mm
- R² Score: 0.1161

These are regression evaluation metrics, not accuracy percentages. The model has limitations in predicting heavy rainfall.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- HTML, CSS and JavaScript
- SQLite

## Project Structure

- `main.py` — FastAPI application entry point
- `auth.py` — Authentication functionality
- `database.py` — Database configuration
- `models.py` — Database models
- `prediction.py` — Rainfall prediction functionality
- `data/` — Historical weather dataset
- `outputs/` — Trained model and evaluation outputs
- `templates/` — HTML templates
- `static/` — CSS and JavaScript files
- `requirements.txt` — Python dependencies

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/abdulhaseebmemon06-ai/MonsoonWatch-Pakistan.git
cd MonsoonWatch-Pakistan
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m uvicorn main:app --reload
```

Open your browser at:

`http://127.0.0.1:8000`

## Important Limitations

The model predicts next-day rainfall, not actual flood events. It is an academic demonstration and not an official weather-warning or flood-prediction service. Prediction results depend on the accuracy of the weather observations provided.

## Developer

**Abdul Haseeb Memon**  
Roll No: 24BSIT-25  
Department of Information Technology  
Shaheed Benazir Bhutto University, Shaheed Benazirabad
