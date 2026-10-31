
import pandas as pd
from pathlib import Path

# Project folder
project_dir = Path(__file__).resolve().parent.parent
results_path = project_dir / "results" / "model_comparison.csv"

# Results from previous 5-fold cross-validation
model_results = [
    {
        "Model": "Linear Regression",
        "Features": "7 Numerical",
        "Mean_RMSE": 54031.99,
        "Std_RMSE": 38495.27,
        "Mean_MAE": 22258.18,
        "Mean_R2": 0.3201
    },
    {
        "Model": "Huber Regression",
        "Features": "7 Numerical",
        "Mean_RMSE": 59908.65,
        "Std_RMSE": 50473.78,
        "Mean_MAE": 21863.14,
        "Mean_R2": 0.0605
    },
    {
        "Model": "Ridge Regression",
        "Features": "7 Numerical",
        "Mean_RMSE": 53429.25,
        "Std_RMSE": 37244.03,
        "Mean_MAE": 22211.07,
        "Mean_R2": 0.3439
    },
    {
        "Model": "Random Forest",
        "Features": "7 Numerical",
        "Mean_RMSE": 33783.10,
        "Std_RMSE": 4738.90,
        "Mean_MAE": 20911.80,
        "Mean_R2": 0.8055
    },
    {
        "Model": "Random Forest",
        "Features": "7 Numerical + 5 Categorical",
        "Mean_RMSE": 33381.19,
        "Std_RMSE": 4759.61,
        "Mean_MAE": 20196.80,
        "Mean_R2": 0.8103
    }
]

# Create comparison table
comparison = pd.DataFrame(model_results)

# Sort models by RMSE
comparison = comparison.sort_values("Mean_RMSE")

# Save results
comparison.to_csv(results_path, index=False)

print("HOUSE PRICES - MODEL COMPARISON SUMMARY")
print("=" * 60)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nModel comparison saved successfully!")
print("File location:", results_path)
print("Models compared:", len(comparison))
