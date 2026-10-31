
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

# Project paths
project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "artifacts" / "house_price_model.joblib"
data_path = project_dir / "data" / "test.csv"

print("HOUSE PRICES - LOAD AND TEST SAVED MODEL")
print("=" * 55)

# Load the saved model
model = load(model_path)

print("\nModel loaded successfully!")

# Load Kaggle test data
df = pd.read_csv(data_path)

# Use the same features as training
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "Neighborhood",
    "KitchenQual",
    "SaleCondition",
    "HouseStyle",
    "ExterQual"
]

# Select the first five houses
sample = df[features].head(5)

# Predict log prices and convert back to dollars
log_predictions = model.predict(sample)
predictions = np.expm1(log_predictions)

# Show prediction results
results = pd.DataFrame({
    "Id": df["Id"].head(5).values,
    "PredictedPrice": predictions
})

print("\nPREDICTION RESULTS")
print(results.round(2).to_string(index=False))

print("\nNumber of predictions:", len(predictions))
