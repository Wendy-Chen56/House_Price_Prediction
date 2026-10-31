
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - LARGE HOUSE ANALYSIS")
print("=" * 55)

# Calculate sale price per square foot
df["PricePerSqFt"] = df["SalePrice"] / df["GrLivArea"]

# Look at the 10 largest houses
largest_houses = df.nlargest(10, "GrLivArea")

columns = [
    "Id",
    "OverallQual",
    "GrLivArea",
    "SalePrice",
    "PricePerSqFt",
    "SaleCondition"
]

print("\nTOP 10 LARGEST HOUSES")
print(
    largest_houses[columns]
    .round(2)
    .to_string(index=False)
)

# Compare House 1299 with the full dataset
house = df[df["Id"] == 1299]

print("\nHOUSE ID 1299")
print("Price per square foot:",
      f"${house['PricePerSqFt'].iloc[0]:,.2f}")

print("\nDATASET MEDIAN")
print("Median price per square foot:",
      f"${df['PricePerSqFt'].median():,.2f}")
