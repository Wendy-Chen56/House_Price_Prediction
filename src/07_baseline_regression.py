
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

print("HOUSE PRICES - BASELINE LINEAR REGRESSION")
print("=" * 55)

# Select features and target
features = ["OverallQual", "GrLivArea"]

X = df[features]
y = df["SalePrice"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train the baseline model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nFeatures used:")
print(features)

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
