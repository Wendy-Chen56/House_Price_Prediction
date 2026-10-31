
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

print("HOUSE PRICES - DATA QUALITY CHECK")
print("=" * 50)

# Check data types
print("\nData types:")
print(df.dtypes.value_counts())

# Check missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nColumns with missing values:")
print(missing)

# Check duplicate records
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check target variable
print("\nMissing SalePrice values:")
print(df["SalePrice"].isnull().sum())
