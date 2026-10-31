# Project Success Measures

## How Will I Measure Success?

I want to know how close my predicted house prices are to the actual sale prices.

I will mainly use three measures:

- **MAE:** This tells me how much my predictions are off by on average, in dollars.
- **RMSE:** This helps me see if my model makes some really large prediction errors.
- **R²:** This tells me how much of the difference in house prices my model can explain.

## Comparing Models

I started with a simple Linear Regression model using house quality and living area.

After that, I tried adding more house features and testing different models, including Ridge Regression, Huber Regression, and Random Forest.

So far, Random Forest has performed the best in my cross-validation results. Its average MAE was around $20,912, compared with about $22,258 for Linear Regression.

## What Would Make This Project Successful?

For me, success means finding a model that makes reasonably accurate predictions and does not make too many large mistakes.

I also want to understand which house features the model uses most and why some houses are harder to predict.

If the model works well, it could give sellers and real estate agents another reference when estimating house prices.

## What Still Needs Improvement?

I found that some unusual houses are difficult to predict. Even though Random Forest performed better overall, I still need to test the final model and check its limitations.

I would not recommend using the model for real pricing decisions without further testing on more recent and local housing data.
