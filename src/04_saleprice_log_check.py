
import pandas as pd
import numpy as np
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

# Original sale prices
original_price = df["SalePrice"]

# Log-transformed sale prices
log_price = np.log1p(original_price)

print("HOUSE PRICES - LOG TRANSFORMATION CHECK")
print("=" * 50)

print("\nOriginal SalePrice skewness:")
print(round(original_price.skew(), 4))

print("\nLog-transformed SalePrice skewness:")
print(round(log_price.skew(), 4))

print("\nOriginal SalePrice standard deviation:")
print(round(original_price.std(), 2))

print("\nLog-transformed SalePrice standard deviation:")
print(round(log_price.std(), 4))
