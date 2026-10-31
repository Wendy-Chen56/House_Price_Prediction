
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - UNUSUAL HOUSE INSPECTION")
print("=" * 55)

# Find House ID 1299
house = df.loc[df["Id"] == 1299]

# Select important property and sale information
columns = [
    "Id",
    "SalePrice",
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "LotArea",
    "Neighborhood",
    "YearBuilt",
    "YearRemodAdd",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "SaleType",
    "SaleCondition"
]

print("\nHOUSE ID 1299 - PROPERTY DETAILS")
print(house[columns].T.to_string(header=False))

# Compare with the overall dataset
print("\nDATASET COMPARISON")

print("Median SalePrice:",
      f"${df['SalePrice'].median():,.2f}")

print("Median GrLivArea:",
      f"{df['GrLivArea'].median():,.0f} sqft")

print("Maximum GrLivArea:",
      f"{df['GrLivArea'].max():,.0f} sqft")

print("\nHouse ID 1299 found:", len(house))
