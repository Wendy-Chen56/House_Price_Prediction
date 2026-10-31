
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - SALE CONDITION ANALYSIS")
print("=" * 55)

# Compare house prices by sale condition
summary = df.groupby("SaleCondition")["SalePrice"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nSale price by sale condition:")
print(summary.round(2).to_string())

# Check houses with Partial sale condition
partial_houses = df[df["SaleCondition"] == "Partial"]

print("\nPARTIAL SALE CONDITION")
print("Number of houses:", len(partial_houses))

print("\nHouse ID 1299 compared with other Partial houses:")
print(
    partial_houses[
        ["Id", "OverallQual", "GrLivArea", "SalePrice"]
    ].sort_values("SalePrice").head(10).to_string(index=False)
)
