
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - RANDOM FOREST FEATURE COMPARISON")
print("=" * 60)

numerical_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd"
]

categorical_features = [
    "Neighborhood",
    "KitchenQual",
    "SaleCondition",
    "HouseStyle",
    "ExterQual"
]

all_features = numerical_features + categorical_features

X = df[all_features]
y = df["SalePrice"]

# Keep the original test set separate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compare two feature groups
feature_groups = {
    "Numerical Only": numerical_features,
    "Numerical + Categorical": all_features
}

for group_name, selected_features in feature_groups.items():

    print(f"\nMODEL: {group_name}")

    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X_train), start=1
    ):
        X_fold_train = X_train.iloc[train_idx][selected_features]
        X_fold_val = X_train.iloc[val_idx][selected_features]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Build preprocessing for each training fold
        if group_name == "Numerical Only":
            preprocessor = "passthrough"
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numerical_features),
                    ("cat", OneHotEncoder(
                        handle_unknown="ignore"
                    ), categorical_features)
                ]
            )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Train with log-transformed prices
        model.fit(X_fold_train, np.log1p(y_fold_train))

        # Convert predictions back to dollars
        predictions = np.expm1(model.predict(X_fold_val))

        rmse = np.sqrt(
            mean_squared_error(y_fold_val, predictions)
        )
        mae = mean_absolute_error(y_fold_val, predictions)
        r2 = r2_score(y_fold_val, predictions)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(
            f"Fold {fold}: "
            f"RMSE=${rmse:,.2f}, "
            f"MAE=${mae:,.2f}, "
            f"R²={r2:.4f}"
        )

    print("\nCROSS-VALIDATION SUMMARY")
    print(f"Mean RMSE: ${np.mean(rmse_scores):,.2f}")
    print(f"Std RMSE:  ${np.std(rmse_scores):,.2f}")
    print(f"Mean MAE:  ${np.mean(mae_scores):,.2f}")
    print(f"Mean R²:   {np.mean(r2_scores):.4f}")

print("\nReserved test records:", len(X_test))
