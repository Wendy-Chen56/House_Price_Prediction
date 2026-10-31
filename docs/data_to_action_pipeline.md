# Data-to-Action Pipeline

## Project: House Price Prediction

### 1. Data Collection — Completed

I downloaded the House Prices dataset from Kaggle. It includes house information such as size, quality, age, neighborhood, and sale price.

**Tool:** Kaggle

### 2. Data Preparation — Completed for Selected Features

I checked the data for missing values and selected seven numerical features and five categorical features. I used One-Hot Encoding to prepare the categorical information for the model.

**Tool:** Python, pandas, scikit-learn

### 3. Model Training — Completed

I tested Linear Regression, Ridge Regression, Huber Regression, and Random Forest. Random Forest performed the best in my cross-validation comparison.

**Tool:** Python, scikit-learn

### 4. Model Evaluation — Completed

I compared the models using RMSE, MAE, and R². I also checked unusual houses that caused large prediction errors.

**Tool:** Python

### 5. Save and Reload Model — Completed

I saved my Random Forest model as a joblib file and tested it in a new Python environment. I was able to reproduce the same predictions and evaluation results.

**Tool:** joblib, GitHub

### 6. Price Prediction — Completed as a Prototype

The saved model can take house information and predict a sale price. I tested it using five houses from the Kaggle test dataset.

**Tool:** Python, Random Forest

### 7. Pricing Decision — Planned

A home seller or real estate agent could compare the predicted price with other market information before deciding on a listing price.

**User:** Home sellers and real estate agents

### 8. Feedback and Model Improvement — Planned

In the future, I would like to compare predictions with actual sale prices, check large errors, and update the model using newer local housing data.
