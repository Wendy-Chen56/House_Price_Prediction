
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Project paths
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "results" / "test_predictions.csv"
output_dir = project_dir / "figures"
output_dir.mkdir(exist_ok=True)

# Load saved test predictions
df = pd.read_csv(data_path)

actual = df["ActualPrice"]
predicted = df["PredictedPrice"]

# Create scatter plot
fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(
    actual,
    predicted,
    alpha=0.55,
    s=35
)

# Perfect prediction reference line
max_price = max(actual.max(), predicted.max())
ax.plot(
    [0, max_price],
    [0, max_price],
    linestyle="--",
    color="red",
    linewidth=1.5,
    label="Perfect Prediction"
)

ax.set_title("Actual vs. Predicted House Prices")
ax.set_xlabel("Actual Sale Price ($)")
ax.set_ylabel("Predicted Sale Price ($)")
ax.set_xlim(0, max_price * 1.05)
ax.set_ylim(0, max_price * 1.05)
ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

# Save figure
output_path = output_dir / "actual_vs_predicted.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("ACTUAL VS. PREDICTED PRICES")
print("=" * 45)
print(f"Total test houses: {len(df)}")
print(f"Figure saved to: {output_path}")
