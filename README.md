# House Price Prediction - MSBA 265 Final Project

## About This Project

This project explores how well machine learning can predict house prices using property information.

The House Prices dataset from Kaggle was used to compare several regression models. The project also examines which house features are related to sale prices and why some houses are harder to predict.

A FastAPI application was added so users can enter house information and receive a predicted sale price using the saved Random Forest model.

## Dataset

The dataset comes from Ames, Iowa, and includes information such as house size, quality, age, neighborhood, and sale price.

The training data contains 1,460 houses and 81 columns, including the house ID, 79 explanatory variables, and SalePrice.

Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

The original Kaggle CSV files are not included in this public GitHub repository.

For complete download instructions, see [data/README.md](data/README.md).

The saved Random Forest model is included in the `artifacts/` folder. Users who only want to test the FastAPI application do not need to retrain the model or download the original dataset.

### How to Download the Dataset

1. Open the Kaggle dataset link above.
2. Sign in to Kaggle or create a free account.
3. Join the competition and accept its rules if required.
4. Open the Data tab and download the competition files.
5. Extract the downloaded ZIP file.
6. Copy `train.csv` and `test.csv` into the `data/` folder of this project.

The folder should look like this:

```text
House_Price_Prediction/
    data/
        train.csv
        test.csv
        data_description.txt
        README.md
```

Do not rename the CSV files.

Additional information is available in `data/README.md`.

The saved Random Forest model is included in the `artifacts/` folder. Users who only want to test the FastAPI application do not need to retrain the model or download the original dataset.

## Models Tested

The following regression models were compared:

- Linear Regression
- Ridge Regression
- Huber Regression
- Random Forest

Categorical features, including neighborhood and kitchen quality, were also tested to see whether they improved prediction performance.

Random Forest with 12 original features had the lowest average RMSE in the five-fold cross-validation comparison.

## Results

The final Random Forest model uses seven numerical features and five categorical features.

The model was trained using 1,168 houses and evaluated on a test set of 292 houses.

### Final Test Results

| Metric | Result |
|---|---|
| RMSE | $29,971.00 |
| MAE | $18,112.09 |
| R² | 0.8829 |

The results were encouraging, but some unusual houses had large prediction errors.

These results are from the original project evaluation. Results may vary if the data, training split, or model settings are changed.

## Project Files

- `P1/` - First project submission and supporting materials
- `data/` - Dataset instructions and data description
- `src/` - Python scripts for data analysis, model training, and evaluation
- `docs/` - Project documentation, data pipeline description, and API testing results
- `results/` - Model comparison and prediction results
- `artifacts/` - Saved Random Forest model
- `figures/` - Python charts used in the final report
- `report/` - Final report in Markdown, HTML, and PDF formats
- `api/` - FastAPI house price prediction application
- `tests/` - Automated API tests
- `requirements.txt` - Python packages used in this project

## Project Stages

This repository contains the House Price Prediction project and its supporting materials.

The project is organized into three submission stages:

### P1 – Project Proposal and Initial Analysis

P1 focuses on the business problem, dataset, initial data exploration, and baseline regression model.

See [P1/README.md](P1/README.md) for instructions on reviewing and reproducing the P1 results.

The final Random Forest model and FastAPI application are not required for P1 verification.

### P2 – Model Development and Evaluation

P2 focuses on data preparation, model comparison, evaluation, saved model testing, and reproducibility.

The P2 folder will contain the draft report, presentation, and stage-specific verification instructions.

### Final – Complete Project Delivery

The final stage includes the completed report, presentation, model, and supporting analysis.

The FastAPI application demonstrates how the saved model can be used to estimate house prices.

The full setup and execution instructions are provided below.


## How to Run the Project

The project was developed using Python 3.13.9 and Visual Studio Code (VS Code) on Windows.

The instructions below explain how to download, set up, and run the project.

### Step 1: Install the Required Software

Before starting, install:

- Python: https://www.python.org/downloads/
- VS Code: https://code.visualstudio.com/
- Git: https://git-scm.com/downloads

Python 3.13.9 was used during development. Using the same version is recommended.

During Python installation, select the option to add Python to PATH if it is available.

To confirm that Python is installed, open PowerShell or the VS Code terminal and run:

```powershell
python --version
```

The terminal should display the installed Python version.

### Step 2: Download the Project from GitHub

Open VS Code and select Terminal > New Terminal.

Run:

```powershell
git clone https://github.com/Wendy-Chen56/House_Price_Prediction.git
```

Move into the project folder:

```powershell
cd House_Price_Prediction
```

If Git is not installed, the project can also be downloaded manually:

1. Open https://github.com/Wendy-Chen56/House_Price_Prediction
2. Click Code.
3. Select Download ZIP.
4. Extract the ZIP file.
5. Open the extracted project folder in VS Code.

Make sure the terminal is opened in the main project folder before running the Python scripts.

### Step 3: Create a Virtual Environment

From the main project folder, run:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

When the environment is active, the terminal should normally show `(venv)` before the command prompt.

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install the Required Packages

Run:

```powershell
python -m pip install -r requirements.txt
```

This installs the Python packages used in the project.

Wait until the installation finishes without errors before continuing.

### Step 5: Set Up the Kaggle Dataset

To reproduce the data analysis or retrain the model, follow the download and setup instructions in [data/README.md](data/README.md).

Make sure `data/train.csv` and `data/test.csv` are in the correct folder before running the analysis scripts.

If you only want to test the saved model through FastAPI, you can skip this step and go directly to the FastAPI section.


### Step 6: Train and Save the Model

From the main project folder, run:

```powershell
python src/24_save_model.py
```

This script trains and saves the Random Forest model.

The saved model is stored in the `artifacts/` folder.

### Step 7: Load the Saved Model and Make Predictions

Run:

```powershell
python src/25_load_test_model.py
```

This script loads the saved Random Forest model and tests its prediction functionality.

The model predicts the log-transformed sale price, so `np.expm1()` is used to convert predictions back into dollars.

### Step 8: Evaluate the Final Model

Run:

```powershell
python src/26_final_model_evaluation.py
```

This script evaluates the final Random Forest model.

The original project reported the following test results:

| Metric | Result |
|---|---|
| RMSE | $29,971.00 |
| MAE | $18,112.09 |
| R² | 0.8829 |

### Step 9: Create the Model Comparison Summary

Run:

```powershell
python src/27_model_comparison_summary.py
```

This script summarizes the results from the regression models tested in the project.

The models include Linear Regression, Ridge Regression, Huber Regression, and Random Forest.

## Final Report and Figures

The final report includes exploratory data analysis, model comparisons, evaluation results, business value, and limitations.

The report files are available in the `report/` folder:

- `final_report.md` - Markdown version
- `final_report.html` - HTML version
- `final_report_numbered.pdf` - Final PDF report with page numbers

Eight figures were created using Python, Matplotlib, and Seaborn. They are stored in the `figures/` folder.

The figures include:

- Distribution of residential sale prices
- Relationship between living area and sale price
- Average sale price by housing quality
- Feature correlation heatmap
- Model comparison results
- Prediction error distribution
- Actual versus predicted sale prices
- Random Forest feature importance

The existing report and figures can be reviewed without rerunning the analysis.

## FastAPI House Price Prediction

The project includes a FastAPI application that uses the saved Random Forest model to predict house prices.

The API accepts 12 house features and returns an estimated sale price in US dollars.

### Step 1: Start the API

Make sure the virtual environment is active and the required packages are installed.

From the main project folder, run:

```powershell
python -m uvicorn api.main:app --reload
```

The terminal should show that the application server is running.

Keep the terminal open while using the API.

### Step 2: Open the API Documentation

Open a browser and visit:

http://127.0.0.1:8000/docs

This opens the FastAPI interactive documentation page.

### Step 3: Check the API Health

1. Find the `GET /health` endpoint.
2. Click to expand it.
3. Click Try it out.
4. Click Execute.
5. Check the response.

A successful health check should return HTTP status 200.

### Step 4: Make a House Price Prediction

1. Find the `POST /predict` endpoint.
2. Click to expand it.
3. Click Try it out.
4. Replace the request body with the example below.
5. Click Execute.
6. Review the predicted sale price in the response body.

### Example Request

```json
{
  "OverallQual": 7,
  "GrLivArea": 1800,
  "GarageCars": 2,
  "TotalBsmtSF": 1000,
  "FullBath": 2,
  "YearBuilt": 2005,
  "YearRemodAdd": 2005,
  "Neighborhood": "NAmes",
  "KitchenQual": "Gd",
  "SaleCondition": "Normal",
  "HouseStyle": "1Story",
  "ExterQual": "Gd"
}
```

### Example Response

```json
{
  "predicted_sale_price_usd": 220642.90,
  "model": "Random Forest",
  "currency": "USD"
}
```

This example uses sample house information. The predicted price is not compared with an actual sale price.

The exact prediction may change if the model is retrained.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks whether the API and saved model are available |
| `/predict` | POST | Predicts a house price using 12 input features |

### Step 5: Stop the API

Return to the terminal where the API server is running.

Press Ctrl + C to stop the server.

## API Testing

The API was tested using the FastAPI documentation page and automated tests.

The manual test confirmed that the API returned the same prediction as running the saved model directly in Python.

An invalid input test was also performed by setting `OverallQual` to 15. The API correctly rejected the request with HTTP status 422.

### Run Automated Tests

From the main project folder, run:

```powershell
python -m pytest tests/test_api.py -v
```

### Original Test Results

| Test | Result |
|---|---|
| Health Check | PASSED |
| House Price Prediction | PASSED |
| Invalid Input Validation | PASSED |

All three automated tests passed during the original project testing.

Additional details are available in `docs/fastapi_validation.md`.

## Troubleshooting

### Python Is Not Recognized

Make sure Python is installed and added to PATH.

Check the installed version:

```powershell
python --version
```

On some Windows computers, `py` can be used instead of `python`.

### PowerShell Blocks Virtual Environment Activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### ModuleNotFoundError

Make sure the virtual environment is active.

Install the required packages again:

```powershell
python -m pip install -r requirements.txt
```

### FileNotFoundError for train.csv

Make sure `train.csv` is located inside the `data/` folder.

Download the Kaggle dataset separately if needed.

Also check that the terminal is opened in the main project folder.

### Saved Model Cannot Be Found

Check the `artifacts/` folder for the saved Random Forest model.

If the model file is missing, download the Kaggle dataset and run:

```powershell
python src/24_save_model.py
```

### FastAPI Documentation Page Does Not Open

Make sure the API server is running:

```powershell
python -m uvicorn api.main:app --reload
```

Then open:

http://127.0.0.1:8000/docs

If the terminal shows an error, resolve that error before trying the browser again.

## Limitations

The dataset contains historical house sales from Ames, Iowa. The model should not be used to estimate current house prices in other cities without additional testing.

Some unusual houses were difficult to predict. These records were kept in the dataset instead of being removed just to improve the model scores.

The test set was also examined during earlier model development, so the final results should not be considered completely independent validation.

The FastAPI tests confirm that the application works and returns consistent predictions. They do not establish prediction accuracy on new real-world housing data.

## Possible Use

This project could provide home sellers and real estate agents with another reference when estimating house prices.

The predicted price can be compared with recent sales of similar properties before making a pricing decision.

The model is not meant to replace a professional appraisal or local market knowledge.

Future improvements could include testing with newer housing data, evaluating performance in other cities, and collecting feedback from real estate professionals.
