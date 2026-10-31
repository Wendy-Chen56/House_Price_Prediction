
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set project paths
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"
output_dir = project_dir / "figures"
output_dir.mkdir(exist_ok=True)

# Load the housing dataset
df = pd.read_csv(data_path)

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(df["SalePrice"], bins=30, edgecolor="black")

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Save the figure
output_path = output_dir / "saleprice_distribution.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("SALE PRICE DISTRIBUTION")
print("=" * 45)
print(f"Total houses: {len(df)}")
print(f"Mean price: ${df['SalePrice'].mean():,.2f}")
print(f"Median price: ${df['SalePrice'].median():,.2f}")
print(f"Minimum price: ${df['SalePrice'].min():,.2f}")
print(f"Maximum price: ${df['SalePrice'].max():,.2f}")
print(f"\nFigure saved to: {output_path}")
