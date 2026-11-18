import pandas as pd
from pathlib import Path
from pathlib import Path

from fastapi import (
    FastAPI,
    Request,
    Depends,
    Form,
    HTTPException
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database import Base, engine, get_db
from models import User
from auth import hash_password, verify_password

# Rainfall prediction functions
from prediction import (
    get_provinces,
    get_locations,
    predict_next_day_rainfall
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "pakistan_flood_locations_daily_weather_1981_2026.csv"

historical_df = pd.read_csv(DATA_PATH)

historical_df["date"] = pd.to_datetime(historical_df["date"])

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="MonsoonWatch Pakistan",
    description="Pakistan Monsoon Rainfall Forecasting System",
    version="1.0.0"
)

# Replace this with a long random secret before deployment.
app.add_middleware(
    SessionMiddleware,
    secret_key="CHANGE-THIS-TO-A-LONG-RANDOM-SECRET-KEY"
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# 2. HOME ROUTE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    if request.session.get("user_id"):
        return RedirectResponse("/dashboard", status_code=303)

    return RedirectResponse("/login", status_code=303)


# --------------------------------------------------
# 3. REGISTRATION
# --------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"error": None}
    )


@app.post("/register")
def register(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):

    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name or len(full_name) > 100:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Please enter a valid full name."},
            status_code=400
        )

    if len(password) < 8 or len(password.encode("utf-8")) > 72:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error": "Password must be at least 8 characters and no more than 72 bytes."
            },
            status_code=400
        )

    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "This email is already registered."},
            status_code=400
        )

    user = User(
        full_name=full_name,
        email=email,
        password_hash=hash_password(password)
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)

    except IntegrityError:
        db.rollback()

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "This email is already registered."},
            status_code=400
        )

    request.session.clear()
    request.session["user_id"] = user.id

    return RedirectResponse(
        "/dashboard",
        status_code=303
    )


# --------------------------------------------------
# 4. LOGIN
# --------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None}
    )


@app.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):

    email = email.strip().lower()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user or not verify_password(
        password,
        user.password_hash
    ):

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Invalid email or password."},
            status_code=401
        )

    request.session.clear()
    request.session["user_id"] = user.id

    return RedirectResponse(
        "/dashboard",
        status_code=303
    )


# --------------------------------------------------
# 5. DASHBOARD
# --------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login", status_code=303)

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"user": user}
    )


# --------------------------------------------------
# 6. RAINFALL PREDICTION PAGE
# --------------------------------------------------

@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(
    request: Request,
    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login", status_code=303)

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={
            "user": user,
            "provinces": get_provinces(),
            "locations": [],
            "error": None,
            "result": None
        }
    )


# --------------------------------------------------
# 7. PROCESS RAINFALL PREDICTION
# --------------------------------------------------

@app.post("/prediction", response_class=HTMLResponse)
def prediction_submit(
    request: Request,

    observation_date: str = Form(...),
    province: str = Form(...),
    location: str = Form(...),

    rainfall_mm: float = Form(...),
    temperature_c: float = Form(...),
    max_temperature_c: float = Form(...),
    min_temperature_c: float = Form(...),
    relative_humidity_percent: float = Form(...),
    wind_speed_mps: float = Form(...),

    rainfall_previous_day_mm: float = Form(...),
    rainfall_previous_3_days_mm: float = Form(...),
    rainfall_previous_7_days_mm: float = Form(...),
    rainfall_previous_7_day_average_mm: float = Form(...),

    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login", status_code=303)

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    # Collect all prediction inputs
    inputs = {
        "observation_date": observation_date,
        "province": province,
        "location": location,

        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "max_temperature_c": max_temperature_c,
        "min_temperature_c": min_temperature_c,
        "relative_humidity_percent": relative_humidity_percent,
        "wind_speed_mps": wind_speed_mps,

        "rainfall_previous_day_mm": rainfall_previous_day_mm,
        "rainfall_previous_3_days_mm": rainfall_previous_3_days_mm,
        "rainfall_previous_7_days_mm": rainfall_previous_7_days_mm,
        "rainfall_previous_7_day_average_mm": rainfall_previous_7_day_average_mm
    }

    # Generate prediction
    try:

        result = predict_next_day_rainfall(inputs)
        error = None

    except (ValueError, TypeError, KeyError) as exc:

        result = None
        error = str(exc)

    except Exception:

        result = None
        error = "Prediction could not be generated. Please check your inputs."

    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={
            "user": user,
            "provinces": get_provinces(),
            "locations": get_locations(province),
            "selected_province": province,
            "selected_location": location,
            "error": error,
            "result": result
        }
    )


# --------------------------------------------------
# 8. PROVINCE-WISE LOCATION API
# --------------------------------------------------

@app.get("/api/locations/{province}")
def locations_api(
    request: Request,
    province: str
):

    if not request.session.get("user_id"):
        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    if province not in get_provinces():
        raise HTTPException(
            status_code=404,
            detail="Province not found."
        )

    return {
        "locations": get_locations(province)
    }


# --------------------------------------------------
# 9. LOGOUT
# --------------------------------------------------

@app.get("/logout")
def logout(request: Request):

    request.session.clear()

    return RedirectResponse(
        "/login",
        status_code=303
    )


# --------------------------------------------------
# 10. API HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "success",
        "application": "MonsoonWatch Pakistan",
        "message": "FastAPI backend is running"
    }


# --------------------------------------------------
# HISTORICAL ANALYSIS PAGE
# --------------------------------------------------

@app.get("/historical", response_class=HTMLResponse)
async def historical_analysis(
    request: Request,
    db: Session = Depends(get_db)
):
    # Check user session
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    # Retrieve logged-in user
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # Dataset overview
    total_records = len(historical_df)
    total_locations = historical_df["location"].nunique()
    total_provinces = historical_df["province"].nunique()

    start_date = historical_df["date"].min().strftime("%d %B %Y")
    end_date = historical_df["date"].max().strftime("%d %B %Y")

    # Monthly average rainfall
    monthly_data = (
        historical_df.groupby("month")["rainfall_mm"]
        .mean()
        .round(2)
        .to_dict()
    )

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly_labels = month_names
    monthly_values = [
        monthly_data.get(month, 0)
        for month in range(1, 13)
    ]

    # Yearly total rainfall
    yearly_data = (
        historical_df.groupby("year")["rainfall_mm"]
        .sum()
        .round(2)
    )

    yearly_labels = yearly_data.index.astype(str).tolist()
    yearly_values = yearly_data.values.tolist()

    # Province-wise location counts
    province_data = (
        historical_df.groupby("province")["location"]
        .nunique()
        .to_dict()
    )

    province_labels = list(province_data.keys())
    province_values = list(province_data.values())

    # Render historical analysis page
    return templates.TemplateResponse(
        request=request,
        name="historical.html",
        context={
            "user": user,
            "total_records": total_records,
            "total_locations": total_locations,
            "total_provinces": total_provinces,
            "start_date": start_date,
            "end_date": end_date,
            "monthly_labels": monthly_labels,
            "monthly_values": monthly_values,
            "yearly_labels": yearly_labels,
            "yearly_values": yearly_values,
            "province_labels": province_labels,
            "province_values": province_values
        }
    )





# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------

@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_page(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # Model evaluation results (V2)
    evaluation_data = [
        {
            "model": "Mean Baseline",
            "samples": 73084,
            "mae": 1.9233,
            "rmse": 9.2336,
            "r2": -0.0099
        },
        {
            "model": "Random Forest V2",
            "samples": 73084,
            "mae": 1.8871,
            "rmse": 8.6384,
            "r2": 0.1161
        }
    ]

    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context={
            "user": user,
            "evaluation_data": evaluation_data,
            "mae": 1.8871,
            "rmse": 8.6384,
            "r2": 0.1161,
            "test_samples": 73084,
            "heavy_rain_samples": 1770,
            "heavy_rain_mae": 28.849,
            "heavy_actual_mean": 36.975,
            "heavy_predicted_mean": 8.338
        }
    )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

@app.get("/about", response_class=HTMLResponse)
async def about_page(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "user": user
        }
    )