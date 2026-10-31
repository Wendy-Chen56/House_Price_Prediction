
import pandas as pd
from pathlib import Path

# Load the training data
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"

df = pd.read_csv(data_path)

print("HOUSE PRICES - FEATURE CORRELATION")
print("=" * 50)

# Select numeric columns
numeric_df = df.select_dtypes(include="number")

# Calculate correlation with SalePrice
correlations = numeric_df.corr()["SalePrice"]

# Remove SalePrice itself
correlations = correlations.drop("SalePrice")

# Sort by strongest absolute correlation
correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

print("\nTop 10 features related to SalePrice:")
print(correlations.head(10).round(4))
