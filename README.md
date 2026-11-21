# Electricity Demand Forecasting Using Python

This repository contains the data preparation and exploratory analysis workflow for Spanish electricity demand and weather data.

## Run the interactive forecasting application

The Streamlit app uses `energy_dataset.csv` from this repository and provides:
- Historical hourly electricity demand visualisation
- Random Forest and Histogram Gradient Boosting models
- Chronological holdout evaluation (MAE, RMSE, MAPE)
- Recursive short-term forecasts
- CSV export of forecast results

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Sign in with GitHub and grant access to this repository.
3. Select repository `adeelshah110/Forecasting-time-series-project-2`.
4. Select branch `main` and main file path `app.py`.
5. Click Deploy.

The app downloads the original electricity dataset from this repository at runtime. The weather dataset is retained for the separate data preparation workflow; weather predictors are not yet used by the first forecasting app version.
