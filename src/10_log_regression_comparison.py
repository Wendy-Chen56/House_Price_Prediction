
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - ORIGINAL VS LOG REGRESSION")
print("=" * 55)

# Use the same seven features
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd"
]

X = df[features]
y = df["SalePrice"]

# Use the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Model 1: Original SalePrice
original_model = LinearRegression()
original_model.fit(X_train, y_train)
original_pred = original_model.predict(X_test)

# Model 2: Log-transformed SalePrice
log_model = LinearRegression()
log_model.fit(X_train, np.log1p(y_train))

# Convert log predictions back to dollars
log_pred = np.expm1(log_model.predict(X_test))

# Evaluate both models in dollars
def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

original_results = evaluate_model(y_test, original_pred)
log_results = evaluate_model(y_test, log_pred)

print("\nORIGINAL SALEPRICE MODEL")
print(f"RMSE: ${original_results[0]:,.2f}")
print(f"MAE:  ${original_results[1]:,.2f}")
print(f"R²:   {original_results[2]:.4f}")

print("\nLOG-TRANSFORMED SALEPRICE MODEL")
print(f"RMSE: ${log_results[0]:,.2f}")
print(f"MAE:  ${log_results[1]:,.2f}")
print(f"R²:   {log_results[2]:.4f}")

print("\nMODEL COMPARISON")
print(f"RMSE difference: ${original_results[0] - log_results[0]:,.2f}")
print(f"MAE difference:  ${original_results[1] - log_results[1]:,.2f}")
