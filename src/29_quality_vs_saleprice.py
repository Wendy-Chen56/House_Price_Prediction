
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"
output_dir = project_dir / "figures"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

correlation = df["OverallQual"].corr(df["SalePrice"])

plt.figure(figsize=(10, 6))

plt.scatter(
    df["OverallQual"],
    df["SalePrice"],
    alpha=0.35,
    s=25
)

plt.title("Overall Quality vs. House Sale Price")
plt.xlabel("Overall Quality Rating (1–10)")
plt.ylabel("Sale Price ($)")
plt.xticks(range(1, 11))
plt.grid(alpha=0.3)

plt.text(
    0.03, 0.95,
    f"Correlation: {correlation:.3f}",
    transform=plt.gca().transAxes,
    va="top"
)

plt.tight_layout()

output_path = output_dir / "quality_vs_saleprice.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("OVERALL QUALITY VS. SALE PRICE")
print("=" * 45)
print(f"Correlation: {correlation:.4f}")
print(f"Total houses: {len(df)}")
print(f"Figure saved to: {output_path}")
