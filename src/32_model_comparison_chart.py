
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Project paths
project_dir = Path(__file__).resolve().parent.parent
results_path = project_dir / "results" / "model_comparison.csv"
output_dir = project_dir / "figures"
output_dir.mkdir(exist_ok=True)

# Load actual model results
df = pd.read_csv(results_path)

# Create readable model labels
df["Label"] = df.apply(
    lambda row: f"{row['Model']} ({row['Features']})",
    axis=1
)

# Sort from lowest to highest RMSE
df = df.sort_values("Mean_RMSE")

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.barh(
    df["Label"],
    df["Mean_RMSE"],
    color=[
        "#2A9D8F" if "12" in label or "5 Categorical" in label
        else "#669BBC"
        for label in df["Label"]
    ]
)

# Put the best model at the top
ax.invert_yaxis()

# Add dollar values to bars
for bar, value in zip(bars, df["Mean_RMSE"]):
    ax.text(
        value + 500,
        bar.get_y() + bar.get_height() / 2,
        f"${value:,.0f}",
        va="center"
    )

ax.set_title("Model Comparison: 5-Fold Cross-Validation RMSE")
ax.set_xlabel("Average RMSE ($) — Lower Is Better")
ax.set_ylabel("Model")
ax.set_xlim(0, df["Mean_RMSE"].max() * 1.22)
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

plt.tight_layout()

output_path = output_dir / "model_comparison_rmse.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("MODEL COMPARISON CHART")
print("=" * 45)
print(f"Best model: {df.iloc[0]['Label']}")
print(f"Lowest mean RMSE: ${df.iloc[0]['Mean_RMSE']:,.2f}")
print(f"Figure saved to: {output_path}")
