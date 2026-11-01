from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="House Price Prediction API",
    description="Enter house information to estimate a sale price using a trained machine learning model.",
    version="1.0.0"
)

# Load saved model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "house_price_model.joblib"

model = joblib.load(MODEL_PATH)


# House information submitted by the user
class HouseInput(BaseModel):
    OverallQual: int = Field(ge=1, le=10)
    GrLivArea: float = Field(gt=0)
    GarageCars: int = Field(ge=0)
    TotalBsmtSF: float = Field(ge=0)
    FullBath: int = Field(ge=0)
    YearBuilt: int = Field(ge=1800, le=2100)
    YearRemodAdd: int = Field(ge=1800, le=2100)
    Neighborhood: str
    KitchenQual: str
    SaleCondition: str
    HouseStyle: str
    ExterQual: str


@app.get("/health", summary="Check API and Model Status")
def health_check():
    return {
        "status": "ok",
        "message": "House Price Prediction API is running",
        "model_loaded": True
    }


@app.post("/predict", summary="Predict House Sale Price")
def predict_price(house: HouseInput):
    # Convert input into the same DataFrame format used for training
    input_df = pd.DataFrame([house.model_dump()])

    # Model predicts log-transformed sale price
    log_prediction = model.predict(input_df)[0]

    # Convert back to dollars
    predicted_price = float(np.expm1(log_prediction))

    return {
        "predicted_sale_price_usd": round(predicted_price, 2),
        "model": "Random Forest",
        "currency": "USD"
    }
    