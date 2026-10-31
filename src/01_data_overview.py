
import pandas as pd
from pathlib import Path

# Find the project folder and load the training data
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

# Check the dataset
print("HOUSE PRICES - DATA OVERVIEW")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nNumber of columns:")
print(len(df.columns))

print("\nTarget variable:")
print("SalePrice")
