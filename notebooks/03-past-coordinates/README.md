# implement Ridge Regression with past coordinates

This document is for explaining the details of the implementation with past coordinates.

## Structure
- 01-implement-past-coord/
    - train.py
        - y(t) = x(t)ß + αy(t-k) + b
        - inlcudes training(rolling origin cross validation.
        - use RMSE as the metric to evaluate the performance
    - eval.py
        - use the weigts that correspond to the best performance in train step
        - and make inference and evaluate the results with RMSE
- 02-implement-no-spikes/
    - train.py
        - y(t) = x(t)ß + αy(t-k) + b
        - inlcudes training(rolling origin cross validation.
        - use RMSE as the metric to evaluate the performance
    - eval.py
        - use the weigts that correspond to the best performance in train step
        - and make inference and evaluate the results with RMSE
- 03-analysis.ipynb
    - visualization
    - F-test