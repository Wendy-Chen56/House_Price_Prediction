
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

# Load data
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

# Same split as previous analysis
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):

    if fold == 3:
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = LinearRegression()
        model.fit(X_fold_train, np.log1p(y_fold_train))

        predictions = np.expm1(model.predict(X_fold_val))

        results = X_fold_val.copy()
        results["Id"] = df.loc[X_fold_val.index, "Id"]
        results["ActualPrice"] = y_fold_val.values
        results["PredictedPrice"] = predictions

        results["AbsoluteError"] = abs(
            results["ActualPrice"] - results["PredictedPrice"]
        )

        results = results.sort_values(
            "AbsoluteError", ascending=False
        )

        print("FOLD 3 - LARGEST PREDICTION ERRORS")
        print("=" * 60)

        columns = [
            "Id",
            "OverallQual",
            "GrLivArea",
            "ActualPrice",
            "PredictedPrice",
            "AbsoluteError"
        ]

        print(
            results[columns].head(10).round(2).to_string(index=False)
        )
