
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - 5-FOLD CROSS-VALIDATION")
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

# Keep the final test set separate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create five folds using training data only
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):

    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Train the model using log-transformed prices
    model = LinearRegression()
    model.fit(X_fold_train, np.log1p(y_fold_train))

    # Convert predictions back to dollars
    predictions = np.expm1(model.predict(X_fold_val))

    # Calculate performance in dollars
    rmse = np.sqrt(mean_squared_error(y_fold_val, predictions))
    mae = mean_absolute_error(y_fold_val, predictions)
    r2 = r2_score(y_fold_val, predictions)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    print(f"\nFold {fold}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R²:   {r2:.4f}")

print("\nCROSS-VALIDATION SUMMARY")
print(f"Mean RMSE: ${np.mean(rmse_scores):,.2f}")
print(f"Std RMSE:  ${np.std(rmse_scores):,.2f}")
print(f"Mean MAE:  ${np.mean(mae_scores):,.2f}")
print(f"Mean R²:   {np.mean(r2_scores):.4f}")

print("\nTraining records:", len(X_train))
print("Reserved test records:", len(X_test))
