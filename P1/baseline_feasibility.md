# Baseline Model and Feasibility Analysis

## 1. Purpose

The purpose of the baseline model is to test whether basic house characteristics can be used to estimate sale prices.

A simple Linear Regression model was selected as the starting point before comparing more advanced methods.

## 2. Baseline Model

The baseline model uses two features:

- **OverallQual:** Overall material and finish quality of the house.
- **GrLivArea:** Above-ground living area in square feet.

The target variable is `SalePrice`.

The dataset was divided into 80% training data and 20% testing data using `random_state=42`.

## 3. Baseline Results

| Metric | Result |
|---|---:|
| Training Records | 1,168 |
| Testing Records | 292 |
| RMSE | $44,159.60 |
| MAE | $28,555.25 |
| R² | 0.7458 |

The baseline model achieved an R² of 0.7458 on the test set, indicating that the two selected features explain approximately 74.58% of the variation in house sale prices.

The MAE was $28,555.25, meaning that predictions differed from actual sale prices by approximately $28,555 on average.

The RMSE was $44,159.60, suggesting that some predictions had relatively large errors.

## 4. Feasibility Assessment

The baseline results suggest that machine learning can provide useful information for estimating house prices.

However, using only two features does not capture all the factors that may influence sale prices.

Additional property characteristics and alternative regression models may improve prediction performance.

## 5. Business Relevance

A basic prediction model could provide home sellers and real estate agents with an additional reference when estimating listing prices.

However, the prediction errors are still substantial. The model should not be used as the only basis for setting a house price.

## 6. Next Steps

The next steps are to compare additional regression models, evaluate more property features, and examine unusually large prediction errors.

The project will also use cross-validation to evaluate whether model performance is consistent across different training and validation samples.

## Source

Baseline results generated using `src/07_baseline_regression.py` and the Kaggle House Prices training dataset.
