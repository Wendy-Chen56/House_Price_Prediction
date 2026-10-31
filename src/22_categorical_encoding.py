
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - CATEGORICAL ENCODING")
print("=" * 55)

# Current numerical features
numerical_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd"
]

# New categorical features
categorical_features = [
    "Neighborhood",
    "KitchenQual",
    "SaleCondition",
    "HouseStyle",
    "ExterQual"
]

X = df[numerical_features + categorical_features]
y = df["SalePrice"]

# Split data before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create a preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_features)
    ]
)

# Learn categories from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same encoding to test data
X_test_processed = preprocessor.transform(X_test)

print("\nORIGINAL FEATURES")
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))
print("Total original features:", X.shape[1])

print("\nAFTER ONE-HOT ENCODING")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

print("\nMissing values after preprocessing:")
print(pd.isna(X_train_processed).sum())

print("\nFirst 10 encoded feature names:")
print(preprocessor.get_feature_names_out()[:10])
