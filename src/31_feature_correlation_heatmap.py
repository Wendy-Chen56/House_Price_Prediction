
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "train.csv"
output_dir = project_dir / "figures"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "SalePrice"
]

correlation_matrix = df[features].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True
)

plt.title("Correlation Between House Features and Sale Price")
plt.tight_layout()

output_path = output_dir / "feature_correlation_heatmap.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("FEATURE CORRELATION HEATMAP")
print("=" * 45)
print(correlation_matrix["SalePrice"].sort_values(ascending=False))
print(f"\nFigure saved to: {output_path}")
