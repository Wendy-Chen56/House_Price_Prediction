
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

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

# Keep the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("HOUSE PRICES - OUTLIER SENSITIVITY ANALYSIS")
print("=" * 55)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):

    if fold == 3:
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train the same log regression model
        model = LinearRegression()
        model.fit(X_fold_train, np.log1p(y_fold_train))

        predictions = np.expm1(model.predict(X_fold_val))

        # Evaluation with all Fold 3 houses
        original_rmse = np.sqrt(
            mean_squared_error(y_fold_val, predictions)
        )
        original_mae = mean_absolute_error(
            y_fold_val, predictions
        )
        original_r2 = r2_score(y_fold_val, predictions)

        # Temporarily exclude House ID 1299 from evaluation
        house_ids = df.loc[X_fold_val.index, "Id"]
        mask = (house_ids != 1299).to_numpy()

        filtered_actual = y_fold_val.to_numpy()[mask]
        filtered_predictions = predictions[mask]

        filtered_rmse = np.sqrt(
            mean_squared_error(filtered_actual, filtered_predictions)
        )
        filtered_mae = mean_absolute_error(
            filtered_actual, filtered_predictions
        )
        filtered_r2 = r2_score(
            filtered_actual, filtered_predictions
        )

        print("\nFOLD 3 - ALL HOUSES")
        print(f"RMSE: ${original_rmse:,.2f}")
        print(f"MAE:  ${original_mae:,.2f}")
        print(f"R²:   {original_r2:.4f}")

        print("\nFOLD 3 - EXCLUDING HOUSE ID 1299")
        print(f"RMSE: ${filtered_rmse:,.2f}")
        print(f"MAE:  ${filtered_mae:,.2f}")
        print(f"R²:   {filtered_r2:.4f}")

        print("\nRMSE reduction:")
        print(f"${original_rmse - filtered_rmse:,.2f}")

        print("\nHouses excluded from evaluation:")
        print((~mask).sum())
