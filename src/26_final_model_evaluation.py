
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project paths
project_dir = Path(__file__).resolve().parent.parent

data_path = project_dir / "data" / "train.csv"
model_path = project_dir / "artifacts" / "house_price_model.joblib"

# Load dataset and saved model
df = pd.read_csv(data_path)
model = load(model_path)

print("HOUSE PRICES - SAVED MODEL EVALUATION")
print("=" * 60)

# Same 12 features used during training
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

X = df[features]
y = df["SalePrice"]

# Recreate the original 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Predict using the saved model
log_predictions = model.predict(X_test)
predictions = np.expm1(log_predictions)

# Evaluate predictions in dollars
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL INFORMATION")
print("Model: Random Forest Regression")
print("Features:", len(features))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTEST SET PERFORMANCE")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE:  ${mae:,.2f}")
print(f"R²:   {r2:.4f}")

print("\nPREDICTION EXAMPLES")
results = pd.DataFrame({
    "Id": df.loc[X_test.index, "Id"].values,
    "ActualPrice": y_test.values,
    "PredictedPrice": predictions
})

results["AbsoluteError"] = abs(
    results["ActualPrice"] - results["PredictedPrice"]
)

print(results.head(10).round(2).to_string(index=False))


# Save prediction results to CSV
results_path = project_dir / "results" / "test_predictions.csv"

results.to_csv(results_path, index=False)

print("\nPrediction results saved successfully!")
print("File location:", results_path)
print("Total records saved:", len(results))
