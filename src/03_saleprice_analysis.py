
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

print("HOUSE PRICES - SALE PRICE ANALYSIS")
print("=" * 50)

# Summary statistics for house prices
print("\nSalePrice summary:")
print(df["SalePrice"].describe())

# Median house price
print("\nMedian sale price:")
print(f"${df['SalePrice'].median():,.2f}")

# Skewness
print("\nSalePrice skewness:")
print(round(df["SalePrice"].skew(), 4))
