
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Project paths
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"
model_path = project_dir / "artifacts" / "house_price_model.joblib"

# Load the dataset
df = pd.read_csv(data_path)

print("HOUSE PRICES - SAVE TRAINED MODEL")
print("=" * 55)

# Numerical features
numerical_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd"
]

# Categorical features
categorical_features = [
    "Neighborhood",
    "KitchenQual",
    "SaleCondition",
    "HouseStyle",
    "ExterQual"
]

features = numerical_features + categorical_features

X = df[features]
y = np.log1p(df["SalePrice"])

# Keep the same 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(
            handle_unknown="ignore"
        ), categorical_features)
    ]
)

# Create the full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Train using the training data only
pipeline.fit(X_train, y_train)

# Create artifacts folder if needed
model_path.parent.mkdir(parents=True, exist_ok=True)

# Save the fitted pipeline
dump(pipeline, model_path)

print("\nModel saved successfully!")
print("Model location:", model_path)

print("\nTraining records:", len(X_train))
print("Reserved test records:", len(X_test))
print("Original features:", len(features))
print("Target: log1p(SalePrice)")
