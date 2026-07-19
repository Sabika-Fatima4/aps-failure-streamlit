# APS Failure Prediction -- Streamlit App

This repository contains a Streamlit web application for predicting
**Air Pressure System (APS) failures** in Scania trucks using an
**XGBoost** machine learning model.

## Overview

The project was developed as part of the **GDGoC DataDive'26**
challenge. It combines exploratory data analysis, feature engineering,
cost-sensitive learning, and threshold optimization to identify whether
a truck failure is related to the APS or another subsystem.

## Features

-   XGBoost classifier for APS failure prediction
-   Cost-aware decision threshold optimization
-   Handles missing sensor values through imputation
-   Upload a CSV file containing truck sensor readings
-   Returns APS failure probability and predicted class

## Repository Contents

-   `app.py` -- Streamlit application
-   `xgb_aps_model.pkl` -- Trained model and preprocessing pipeline
-   `requirements.txt` -- Python dependencies

## Running Locally

1.  Clone the repository.
2.  Install dependencies:

``` bash
pip install -r requirements.txt
```

3.  Launch the application:

``` bash
streamlit run app.py
```

The application will open in your browser, where you can upload a CSV
containing truck sensor readings and receive APS failure predictions.

## Model Summary

-   **Algorithm:** XGBoost
-   **Optimized Threshold:** 0.04
-   **Test Business Cost:** 10,390
-   **Test ROC-AUC:** 0.994

The threshold was selected by minimizing the competition's asymmetric
business cost function, where missing an APS failure is significantly
more expensive than generating a false alarm.

## Notes

-   Sensor names are anonymized by the dataset provider.
-   The model is intended as a decision-support tool for maintenance
    triage rather than a replacement for technician expertise.

## Author

**Sabika Fatima**
