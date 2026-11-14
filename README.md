
# Flight Delay Prediction System

## Project Overview

This project uses Machine Learning to predict whether a flight will be delayed by 15 minutes or more.

The system analyzes flight information such as the airline, departure airport, destination airport, flight date, departure time, and flight distance.

A Streamlit dashboard allows users to enter flight details and view the predicted delay risk.

## Dataset

- Source: US flight data from 2024
- Original records: 7,079,081
- Records after cleaning: 6,982,766
- Training records: 250,288
- Final testing records: 106,105

## Data Preparation

The dataset was prepared by:
- Removing cancelled flights
- Handling missing and invalid departure delays
- Creating the target variable (delayed)
- Selecting relevant flight features
- Preparing the data for model training

## Machine Learning Models

Two main models were evaluated:

1. Random Forest
2. Balanced HistGradientBoosting

Random Forest was selected for the application based on its performance on the final test data.

## Final Model Performance

Random Forest was evaluated using flight records from October to December 2024.

| Metric | Result |
|---|---|
| Accuracy | 72.56% |
| Precision | 27.04% |
| Recall | 37.89% |
| F1-Score | 31.56% |

The model can identify some delayed flights, but its performance is limited, especially when predicting future flight patterns.

## Dashboard Features

- Flight information input
- Flight delay prediction
- Estimated delay probability
- Risk level (Low, Moderate, High)
- Model performance summary
- Monthly flight delay visualization

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

## Limitations

The model was trained using 2024 US flight data. Predictions for future dates are estimates and may be less reliable because flight patterns can change.

The application does not provide live flight tracking or official flight status.
