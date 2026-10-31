
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set project paths
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "results" / "test_predictions.csv"
output_dir = project_dir / "figures"
output_dir.mkdir(exist_ok=True)

# Load saved predictions
df = pd.read_csv(data_path)

# Calculate prediction errors in dollars
errors = df["PredictedPrice"] - df["ActualPrice"]

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(errors, bins=30, edgecolor="black", alpha=0.8)

# Zero error reference line
ax.axvline(
    0,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Zero Error"
)

ax.set_title("Distribution of House Price Prediction Errors")
ax.set_xlabel("Prediction Error ($)")
ax.set_ylabel("Number of Houses")
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

output_path = output_dir / "prediction_error_distribution.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("PREDICTION ERROR DISTRIBUTION")
print("=" * 45)
print(f"Total houses: {len(df)}")
print(f"Mean error: ${errors.mean():,.2f}")
print(f"Median error: ${errors.median():,.2f}")
print(f"Mean absolute error: ${errors.abs().mean():,.2f}")
print(f"Figure saved to: {output_path}")
