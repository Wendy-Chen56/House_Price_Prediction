# Reproducibility Test

## Purpose

I wanted to check if my project could run in a new Python environment using only the files from GitHub.

## What I Did

I cloned my GitHub repository into a new folder called `house_prices_reproduction_test`.

I created a new Python virtual environment and installed the packages using `requirements.txt`.

After that, I ran the saved model and evaluation scripts to see if I could get the same results.

## Results

The saved Random Forest model loaded successfully.

I tested it on five houses from the Kaggle test dataset, and the predicted prices matched the results from my original project.

I also ran the model evaluation script again using the 292-house test set.

The results were the same:

- RMSE: $29,971.00
- MAE: $18,112.09
- R²: 0.8829

The script also created `test_predictions.csv` with 292 records.

## What I Learned

This test showed me that my saved model and evaluation code could run in a new environment without using the original project's virtual environment.

I also learned why keeping the Python dependencies and setup instructions is important.

## Next Step

I still need another student to clone the repository, follow my README, and complete the peer verification required by the professor.
