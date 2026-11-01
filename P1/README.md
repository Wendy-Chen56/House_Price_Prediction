# P1 – House Price Prediction

## About P1

This is the first stage of the House Price Prediction project.

P1 focuses on the business problem, dataset, initial data exploration, and a simple baseline regression model.

The goal is to check whether house information can be used to estimate sale prices before comparing more advanced models.

This README explains how to review and reproduce the P1 results.

## 1. Download the Project

Open the public GitHub repository:

https://github.com/Wendy-Chen56/House_Price_Prediction

Click **Code → Download ZIP**, then extract the ZIP file.

Alternatively, use Git:

```powershell
git clone https://github.com/Wendy-Chen56/House_Price_Prediction.git
cd House_Price_Prediction
```

If the project was downloaded as a ZIP, open the extracted `House_Price_Prediction-main` folder instead.

All commands below should be run from the main project folder, not from inside `P1`.

## 2. Set Up Python

The project was developed using Python 3.13.9 on Windows.

Open PowerShell or the VS Code terminal in the main project folder.

Create a virtual environment:

```powershell
python -m venv venv
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

## 3. Download the Dataset

The dataset comes from Kaggle:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The original CSV files are not included in this public repository.

Follow the instructions in `data/README.md` to download the Kaggle files and place them in the `data` folder.

Make sure `data/train.csv` exists before running the P1 Python scripts.

## 4. Review the P1 Submission

Open:

`P1/P1_Submission.pdf`

The PDF contains the topic exploration, infographic, project proposal, initial data exploration, and baseline feasibility analysis.

The individual Markdown files and figures are also available in the `P1` folder.

## 5. Run the Initial Data Exploration

From the main project folder, run:

```powershell
python src/01_data_overview.py
```

This script provides a basic overview of the housing dataset.

Next, run:

```powershell
python src/02_data_quality.py
```

Expected results include:

- 1,460 training records and 81 columns
- 43 object columns, 35 integer columns, and 3 floating-point columns
- 0 duplicate rows
- 0 missing SalePrice values

Some property fields have missing values, including PoolQC, MiscFeature, Alley, Fence, and LotFrontage.

To examine the sale price distribution, run:

```powershell
python src/03_saleprice_analysis.py
```

The P1 report includes the existing figures:

- `P1/saleprice_distribution.png`
- `P1/living_area_vs_saleprice.png`

## 6. Run the Baseline Model

Run:

```powershell
python src/07_baseline_regression.py
```

The baseline uses two features:

- `OverallQual` – overall house quality
- `GrLivArea` – above-ground living area

The target is `SalePrice`.

The data is divided into 1,168 training records and 292 testing records.

Expected model results:

| Metric | Expected Result |
|---|---:|
| RMSE | $44,159.60 |
| MAE | $28,555.25 |
| R² | 0.7458 |

Small differences may occur if package versions differ.

## 7. Review the Findings

The initial exploration showed that the average house price was approximately $180,921, while the median was $163,000.

The sale price distribution was right-skewed, with a skewness value of 1.88.

Living area and sale price had a correlation of 0.7086, indicating a positive relationship.

The baseline model provided a starting point for evaluating whether house characteristics could be used to predict sale prices.

## 8. P1 Verification

A reviewer should be able to:

1. Download or clone the repository.
2. Download the Kaggle dataset using the instructions.
3. Install the Python dependencies.
4. Run the initial data exploration scripts.
5. Run the baseline regression model.
6. Compare the results with `P1/P1_Submission.pdf`.

The Assignment Validation Form is handled separately and is not stored in this public repository.

## 9. Scope of P1

P1 covers the project proposal, initial EDA, and baseline model.

The advanced model comparison, final model evaluation, saved model, and FastAPI application belong to later project stages.

The full project code is available in the repository, but those later components are not required to verify the P1 submission.

## Data Source

Kaggle. *House Prices – Advanced Regression Techniques.*

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
