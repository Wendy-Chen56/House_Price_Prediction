
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Project paths
project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "artifacts" / "house_price_model.joblib"
output_dir = project_dir / "figures"
output_dir.mkdir(exist_ok=True)

# Load saved model
pipeline = joblib.load(model_path)

preprocessor = pipeline.named_steps["preprocessor"]
regressor = pipeline.named_steps["regressor"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()
importances = regressor.feature_importances_

# Check that names match importance values
if len(feature_names) != len(importances):
    raise ValueError("Feature names and importance values do not match.")

# Create feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Keep the 15 most important encoded features
top_features = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(15)
    .sort_values("Importance")
)

# Create chart
fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.set_title("Random Forest Feature Importance (Top 15)")
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Feature")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

plt.tight_layout()

# Save figure
output_path = output_dir / "random_forest_feature_importance.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

# Print results
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 50)
print(f"Total encoded features: {len(feature_names)}")
print("\nTop 15 features:")
print(
    importance_df
    .sort_values("Importance", ascending=False)
    .head(15)
    .to_string(index=False)
)
print(f"\nFigure saved to: {output_path}")
