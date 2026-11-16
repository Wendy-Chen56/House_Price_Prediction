# Business Problem and Project Objective

## Project Title

Predicting House Prices Using Machine Learning

## Business Problem

House prices can be very different, even when houses are in the same area. Some houses are bigger, newer, or in better condition than others. This makes it hard for sellers to decide how much to ask for their homes.

For example, a seller might list a house for $300,000, while a model estimates its price at $260,000. That is a $40,000 difference worth looking into.

If the asking price is too high, the house might take longer to sell. If it is too low, the seller might miss an opportunity to get a better price.

This is just an example, not a measured financial loss.

I want to see if I can use house information to make price estimates instead of just guessing.

## Stakeholders

This project is mainly for home sellers and real estate agents who want some extra information when deciding on a house price.

## Project Question

Can I use machine learning to predict house prices based on things like house size, quality, age, and neighborhood?

## Project Objective

My goal is to compare different regression models and find out which one can predict house prices more accurately.

I want to understand which house features are important and why some houses are harder to predict than others.

## Business Value

This project could help sellers and real estate agents get another estimate when deciding on a listing price.

For example, if the asking price and the model prediction are very different, they could review the property details and compare the house with similar homes before making a decision.

I tested several models and found that Random Forest with 12 features performed the best in my cross-validation comparison.

My final model achieved an RMSE of $29,971, an MAE of $18,112.09, and an R² of 0.8829 on 292 test houses.

I built a working FastAPI prototype to return house price predictions.

However, I have not tested whether using the model actually improves real estate pricing decisions or saves money.

The model would not replace a real estate agent or a professional appraisal.

## Dataset

I am using the House Prices - Advanced Regression Techniques dataset from Kaggle.

The data comes from Ames, Iowa, and includes information such as house size, overall quality, year built, garage size, and sale price.

Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## Project Limitations

One limitation is that the data comes from Ames, Iowa, so the model might not work as well for houses in other cities or in today's housing market.

I noticed that some unusual houses are much harder to predict. For example, one large house in my dataset sold for only $160,000, but my earlier linear model predicted a price over $2 million.

This showed me that even when a model performs well overall, it can still make big mistakes on certain houses.

Before using the model for real pricing decisions, I would need newer local housing data and feedback from real users.