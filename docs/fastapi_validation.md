# FastAPI Validation Test

## Purpose

This test checks whether the FastAPI application can load the saved Random Forest model and return a prediction.

The API prediction is also compared with a prediction made directly in Python to confirm that both methods produce the same result.

## Test Input

The following sample house information was used:

| Feature | Value |
|---|---|
| OverallQual | 7 |
| GrLivArea | 1800 |
| GarageCars | 2 |
| TotalBsmtSF | 1000 |
| FullBath | 2 |
| YearBuilt | 2005 |
| YearRemodAdd | 2005 |
| Neighborhood | NAmes |
| KitchenQual | Gd |
| SaleCondition | Normal |
| HouseStyle | 1Story |
| ExterQual | Gd |

## Test Results

| Test | Result |
|---|---|
| API Endpoint | POST /predict |
| HTTP Status | 200 OK |
| FastAPI Prediction | $220,642.90 |
| Direct Python Prediction | $220,642.90 |
| Results Match | True |

## Invalid Input Test

An additional test was performed to check how the API handles invalid input.

The OverallQual value was changed from 7 to 15, which is outside the allowed range of 1 to 10.

| Test | Result |
|---|---|
| Input | OverallQual = 15 |
| Expected HTTP Status | 422 |
| Actual HTTP Status | 422 |
| Test Result | Passed |

The API rejected the invalid input instead of returning a house price prediction.

## Automated Tests

Three automated tests were run using pytest.

| Test | Result |
|---|---|
| Health Check | PASSED |
| House Price Prediction | PASSED |
| Invalid Input Validation | PASSED |

Test command:

`python -m pytest tests/test_api.py -v`

Results: 3 passed, 1 warning in 4.30 seconds.

The warning came from the FastAPI testing dependencies and did not cause any test failures.

## Conclusion

The FastAPI application successfully loaded the saved Random Forest model and returned a predicted house price of $220,642.90.

The same input was tested directly in Python and produced an identical prediction.

The results confirm that the API is using the saved model correctly and returning consistent predictions.

This test verifies that the API works as expected. It does not measure prediction accuracy because the sample house does not have a known actual sale price.
