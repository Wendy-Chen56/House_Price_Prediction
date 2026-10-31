
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - MULTIPLE LINEAR REGRESSION")
print("=" * 55)

# Use more house features
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

# Keep the same train/test split as the baseline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nMODEL PERFORMANCE")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE:  ${mae:,.2f}")
print(f"R²:   {r2:.4f}")

print("\nMODEL COEFFICIENTS")
print(f"Intercept: {model.intercept_:,.2f}")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:,.2f}")
