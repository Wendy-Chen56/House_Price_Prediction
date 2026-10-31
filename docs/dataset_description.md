# Dataset Description and Data Quality

## Dataset Source

I am using the House Prices - Advanced Regression Techniques dataset from Kaggle.

The data contains house information from Ames, Iowa. Each row represents one house sale.

Dataset link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## Dataset Overview

The training dataset has 1,460 rows and 81 columns.

The main variable I want to predict is `SalePrice`.

The dataset includes information about house size, quality, age, neighborhood, garage, basement, and other property features.

I also have a separate test dataset from Kaggle, but it does not include actual sale prices.

## Data Quality

I checked the data using Python before building my models.

I found:

- 38 numerical columns and 43 string columns.
- No duplicate rows.
- No missing values in `SalePrice`.
- Several columns have missing values.

Some missing values make sense. For example, a missing pool quality value might mean the house does not have a pool.

Other missing values, such as `LotFrontage`, may need to be filled in before using those features in a model.

I don't want to delete rows just because they have missing values without checking what those values mean.

## House Price Summary

The average sale price is around $180,921, while the median is $163,000.

The lowest sale price is $34,900, and the highest is $755,000.

I also found that house prices are right-skewed. After using a log transformation, the skewness dropped from 1.8829 to 0.1213.

This helped me decide to compare regression models using original and log-transformed sale prices.

## Unusual Houses

During my analysis, I found two very large houses with unusually low prices per square foot.

House ID 1299 has 5,642 square feet of living space but sold for only $160,000.

House ID 524 has 4,676 square feet of living space and sold for $184,750.

Both houses have an overall quality rating of 10 and are marked as Partial sales.

These houses are important because they showed me that some properties can be difficult for the model to predict.

I kept them in the dataset rather than removing them just to improve the model scores.

## Data Limitations

The dataset comes from one city and contains historical house sales, so it may not represent current housing prices or other locations.

Also, some property information is missing, and unusual transactions may be harder to predict.

I would need more recent and local housing data before using this model for actual pricing decisions.
