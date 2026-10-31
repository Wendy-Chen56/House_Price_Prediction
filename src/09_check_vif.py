
import pandas as pd
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split

# Load the dataset
project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "train.csv")

print("HOUSE PRICES - MULTICOLLINEARITY CHECK")
print("=" * 55)

# Use the same features as the multiple regression model
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd"
]

X = df[features]

# Use the same training split
X_train, X_test = train_test_split(
    X, test_size=0.20, random_state=42
)

# Add an intercept
X_train = add_constant(X_train)

# Calculate VIF for each feature
vif_data = pd.DataFrame()

vif_data["Feature"] = X_train.columns

vif_data["VIF"] = [
    variance_inflation_factor(X_train.values, i)
    for i in range(X_train.shape[1])
]

# Remove intercept from the results
vif_data = vif_data[vif_data["Feature"] != "const"]

print("\nVariance Inflation Factor (VIF):")
print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))
