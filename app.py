import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px

st.set_page_config(page_title="Spain Electricity Demand Forecasting", page_icon="⚡", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/adeelshah110/Forecasting-time-series-project-2/main/energy_dataset.csv"
TARGET = "total load actual"

@st.cache_data(show_spinner="Loading the electricity dataset...")
def load_data():
    df = pd.read_csv(DATA_URL)
    if "time" not in df.columns or TARGET not in df.columns:
        raise ValueError("The source CSV must contain 'time' and 'total load actual' columns.")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["time", TARGET]).sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return df[["time", TARGET]]

def make_features(series):
    frame = pd.DataFrame({"y": series})
    idx = pd.DatetimeIndex(series.index)
    frame["hour"] = idx.hour
    frame["dayofweek"] = idx.dayofweek
    frame["month"] = idx.month
    frame["dayofyear"] = idx.dayofyear
    frame["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    frame["lag_1"] = frame["y"].shift(1)
    frame["lag_24"] = frame["y"].shift(24)
    frame["lag_168"] = frame["y"].shift(168)
    frame["rolling_mean_24"] = frame["y"].shift(1).rolling(24).mean()
    frame["rolling_mean_168"] = frame["y"].shift(1).rolling(168).mean()
    return frame.dropna()

def build_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42)
    return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=20, l2_regularization=1.0, random_state=42)

def future_features(history, timestamp):
    vals = np.asarray(history, dtype=float)
    return pd.DataFrame([{
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
        "month": timestamp.month,
        "dayofyear": timestamp.dayofyear,
        "is_weekend": int(timestamp.dayofweek >= 5),
        "lag_1": vals[-1],
        "lag_24": vals[-24],
        "lag_168": vals[-168],
        "rolling_mean_24": float(np.mean(vals[-24:])),
        "rolling_mean_168": float(np.mean(vals[-168:]))
    }])

st.title("⚡ Spain Electricity Demand Forecasting")
st.caption("Interactive time-series analysis and short-term forecasting using the Spanish electricity demand dataset.")

try:
    raw = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.sidebar.header("Forecast settings")
model_name = st.sidebar.selectbox("Forecasting model", ["Random Forest", "Histogram Gradient Boosting"])
horizon = st.sidebar.slider("Forecast horizon (hours)", 24, 168, 48, step=24)

series = raw.set_index("time")[TARGET].asfreq("h").interpolate("time").ffill().bfill()
if len(series) < 500:
    st.error("The dataset is too small for the selected lag-based forecasting workflow.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{len(series):,}")
c2.metric("Start date", series.index.min().strftime("%d %b %Y"))
c3.metric("End date", series.index.max().strftime("%d %b %Y"))

st.subheader("Historical electricity demand")
display_days = st.slider("Historical period to display (days)", 7, 90, 30)
history_view = series.tail(display_days * 24).rename("Electricity demand").reset_index()
fig = px.line(history_view, x="time", y="Electricity demand", title="Hourly electricity demand")
fig.update_layout(xaxis_title="Date", yaxis_title="Demand (MW)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

features = make_features(series)
split = int(len(features) * 0.8)
train, test = features.iloc[:split], features.iloc[split:]
X_cols = [c for c in features.columns if c != "y"]
model = build_model(model_name)
with st.spinner("Training the model and evaluating the chronological test period..."):
    model.fit(train[X_cols], train["y"])
    pred = model.predict(test[X_cols])

mae = mean_absolute_error(test["y"], pred)
rmse = float(np.sqrt(mean_squared_error(test["y"], pred)))
denom = np.maximum(np.abs(test["y"].to_numpy()), 1.0)
mape = float(np.mean(np.abs((test["y"].to_numpy() - pred) / denom)) * 100)

st.subheader("Out-of-sample evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:,.1f} MW")
m2.metric("RMSE", f"{rmse:,.1f} MW")
m3.metric("MAPE", f"{mape:.2f}%")

comparison = pd.DataFrame({"Actual": test["y"].tail(24 * 7), "Predicted": pred[-24 * 7:]})
comparison.index.name = "time"
comparison = comparison.reset_index().melt(id_vars="time", var_name="Series", value_name="Demand")
fig_eval = px.line(comparison, x="time", y="Demand", color="Series", title="Actual vs predicted demand — final test week")
fig_eval.update_layout(xaxis_title="Date", yaxis_title="Demand (MW)", hovermode="x unified")
st.plotly_chart(fig_eval, use_container_width=True)

st.subheader(f"Next {horizon} hours forecast")
if len(series) < 168:
    st.warning("At least 168 historical observations are required for weekly lag features.")
else:
    with st.spinner("Generating recursive future forecasts..."):
        full_features = make_features(series)
        final_model = build_model(model_name)
        final_model.fit(full_features[X_cols], full_features["y"])
        history = series.tolist()
        future_times = pd.date_range(series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
        future_values = []
        for ts in future_times:
            row = future_features(history, ts)
            value = float(final_model.predict(row[X_cols])[0])
            future_values.append(value)
            history.append(value)
    future_df = pd.DataFrame({"time": future_times, "Forecast": future_values})
    recent = series.tail(7 * 24).rename("Historical").reset_index()
    chart_data = pd.concat([
        recent.rename(columns={"Historical": "Demand"}).assign(Type="Historical"),
        future_df.rename(columns={"Forecast": "Demand"}).assign(Type="Forecast")
    ], ignore_index=True)
    fig_future = px.line(chart_data, x="time", y="Demand", color="Type", title="Recent demand and future forecast")
    fig_future.update_layout(xaxis_title="Date", yaxis_title="Demand (MW)", hovermode="x unified")
    st.plotly_chart(fig_future, use_container_width=True)
    st.download_button("Download forecast CSV", future_df.to_csv(index=False).encode("utf-8"), "electricity_demand_forecast.csv", "text/csv")

with st.expander("About this application and methodology"):
    st.write("This first deployment uses the electricity demand series and calendar/lag features. Evaluation uses a chronological 80/20 split and one-step-ahead test predictions. Future horizons are generated recursively. Weather variables are not yet included in the forecasting model; they can be added after validating the timestamp merge and feature availability.")
    st.write("The application is an educational/research demonstration, not an operational electricity-grid forecasting service.")

st.caption("Data source: energy_dataset.csv in the associated GitHub repository. Forecast accuracy depends on data quality, model assumptions and forecast horizon.")
