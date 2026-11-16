# Data-to-Action Pipeline

## Project: House Price Prediction

### 1. Data Collection — Completed

I downloaded the House Prices dataset from Kaggle. It includes house information such as size, quality, age, neighborhood, and sale price.

**Tool:** Kaggle

### 2. Data Preparation — Completed

I checked the data for missing values and selected seven numerical features and five categorical features. I used One-Hot Encoding to prepare the categorical information for the model.

**Tool:** Python, pandas, scikit-learn

### 3. Model Training — Completed

I tested Linear Regression, Ridge Regression, Huber Regression, and Random Forest. Random Forest with 12 features performed the best in my cross-validation comparison.

**Tool:** Python, scikit-learn

### 4. Model Evaluation — Completed

I compared the models using RMSE, MAE, and R². My final Random Forest model achieved an RMSE of $29,971, MAE of $18,112.09, and R² of 0.8829 on 292 test houses. I also checked unusual houses that caused large prediction errors.

**Tool:** Python, scikit-learn

### 5. Save and Reload Model — Completed

I saved my Random Forest model as a joblib file and tested it in a separate Python environment. I was able to reproduce the same predictions and evaluation results.

**Tool:** Python, joblib

### 6. Price Prediction — Completed as a Prototype

The saved model can take 12 house features and predict a sale price. I also developed a FastAPI application that allows users to submit house information through an API. All three automated API tests passed.

**Tool:** Python, Random Forest, FastAPI, pytest

### 7. Pricing Decision — Planned

A home seller or real estate agent could compare the predicted price with other market information before deciding on a listing price.

For example, if a seller asks $300,000 and the model predicts $260,000, the $40,000 difference could help start a pricing discussion. This is an example, not a proven financial saving.

**User:** Home sellers and real estate agents

### 8. Feedback and Model Improvement — Planned

In the future, I would like to compare predictions with actual sale prices, check large errors, and update the model using newer local housing data. I would also like to build a more user-friendly web interface.

**Goal:** Improve prediction accuracy and make the model easier to use.