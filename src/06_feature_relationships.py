
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

print("HOUSE PRICES - FEATURE RELATIONSHIPS")
print("=" * 50)

# Select the top 10 numerical features
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd"
]

# Calculate correlations between features
corr_matrix = df[features].corr()

# Find feature pairs with strong correlations
print("\nFeature pairs with correlation above 0.70:")

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        correlation = corr_matrix.iloc[i, j]

        if abs(correlation) >= 0.70:
            print(
                f"{features[i]} and {features[j]}: "
                f"{correlation:.4f}"
            )
