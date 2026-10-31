import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Make the project folder available for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction():
    house = {
        "OverallQual": 7,
        "GrLivArea": 1800,
        "GarageCars": 2,
        "TotalBsmtSF": 1000,
        "FullBath": 2,
        "YearBuilt": 2005,
        "YearRemodAdd": 2005,
        "Neighborhood": "NAmes",
        "KitchenQual": "Gd",
        "SaleCondition": "Normal",
        "HouseStyle": "1Story",
        "ExterQual": "Gd",
    }

    response = client.post("/predict", json=house)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_sale_price_usd"] == 220642.90
    assert result["model"] == "Random Forest"
    assert result["currency"] == "USD"


def test_invalid_quality():
    house = {
        "OverallQual": 15,
        "GrLivArea": 1800,
        "GarageCars": 2,
        "TotalBsmtSF": 1000,
        "FullBath": 2,
        "YearBuilt": 2005,
        "YearRemodAdd": 2005,
        "Neighborhood": "NAmes",
        "KitchenQual": "Gd",
        "SaleCondition": "Normal",
        "HouseStyle": "1Story",
        "ExterQual": "Gd",
    }

    response = client.post("/predict", json=house)

    assert response.status_code == 422
    