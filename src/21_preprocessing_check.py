
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - PREPROCESSING CHECK")
print("=" * 55)

# Check numerical and categorical columns
numeric_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(include=["object", "str"]).columns

print("\nDATA TYPES")
print("Numerical columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# Check missing values by data type
print("\nMISSING VALUES BY DATA TYPE")
print("Numerical missing values:",
      df[numeric_cols].isna().sum().sum())

print("Categorical missing values:",
      df[categorical_cols].isna().sum().sum())

# Check missing values in the seven features already used
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd"
]

print("\nMISSING VALUES IN CURRENT MODEL FEATURES")
print(df[features].isna().sum())

# Check categorical features we may use later
categorical_features = [
    "Neighborhood",
    "KitchenQual",
    "SaleCondition",
    "HouseStyle",
    "ExterQual"
]

print("\nPOTENTIAL CATEGORICAL FEATURES")
print(df[categorical_features].isna().sum())

print("\nUNIQUE VALUES IN POTENTIAL CATEGORICAL FEATURES")
for column in categorical_features:
    print(f"{column}: {df[column].nunique()} categories")
