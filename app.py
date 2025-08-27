import warnings
import numpy as np
import pandas as pd
import requests
import io
import threading
import time

from flask import Flask, jsonify
from flask_cors import CORS

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split

# Configurations
API_URL = "https://api.thingspeak.com/channels/2912101/feeds.csv?api_key=38R3ABOM946O9JZV&results=8000"
TIMEZONE = "Asia/Bangkok"
RESAMPLE_FREQ = "h"
TEST_SIZE = 12
SVR_TIMESTEPS = 5
SVR_TEMP_PARAMS = dict(kernel="rbf", gamma=1, C=10, epsilon=0.10)
SVR_HUMI_PARAMS = dict(kernel="rbf", gamma=0.1, C=10, epsilon=0.05)
UPDATE_INTERVAL = 1800  # 30 minutes

# Display settings
pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings("ignore", category=FutureWarning)

# Shared data containers
latest_df = None
latest_arima_df = None
latest_sarima_df = None
latest_svr_df = None


def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    else:
        raise Exception(f"Failed to retrieve feed. Status code: {response.status_code}")


def dataset_data(feeds):
    feeds.rename(
        columns={"field1": "temperature", "field2": "humidity", "field3": "carbon"},
        inplace=True,
    )
    feeds["timestamp"] = (
        pd.to_datetime(feeds["created_at"]).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    )
    dataset = feeds[["timestamp", "temperature", "humidity"]].copy()
    dataset.set_index("timestamp", inplace=True)
    dataset.ffill(inplace=True)
    return dataset


def preprocess_data(feeds):
    feeds.rename(
        columns={"field1": "temperature", "field2": "humidity", "field3": "carbon"},
        inplace=True,
    )
    feeds["timestamp"] = (
        pd.to_datetime(feeds["created_at"]).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    )
    dataset = feeds[["timestamp", "temperature", "humidity"]].copy()
    dataset.set_index("timestamp", inplace=True)
    dataset.ffill(inplace=True)
    df = dataset.resample(RESAMPLE_FREQ).mean()
    if df.isnull().values.any():
        df.interpolate(method="linear", inplace=True)
    return df


def create_sequences(data, timesteps):
    return np.array([data[i : i + timesteps] for i in range(len(data) - timesteps + 1)])


def train_svr(df, column, timesteps, svr_params):
    train = df.copy()[:-76][[column]]
    test = df.copy()[-76:][[column]]
    scaler = MinMaxScaler()
    train[column] = scaler.fit_transform(train)
    test[column] = scaler.transform(test)
    train_data = train.values
    test_data = test.values
    train_seq = create_sequences(train_data, timesteps)[:, :, 0]
    test_seq = create_sequences(test_data, timesteps)[:, :, 0]
    x_train, y_train = train_seq[:, : timesteps - 1], train_seq[:, [timesteps - 1]]
    x_test, y_test = test_seq[:, : timesteps - 1], test_seq[:, [timesteps - 1]]
    train_timestamps = df[
        (df.index >= train.index[0]) & (df.index < test.index[0])
    ].index[timesteps - 1 :]
    test_timestamps = df[df.index >= test.index[0]].index[timesteps - 1 :]
    svr = SVR(**svr_params)
    svr.fit(x_train, y_train[:, 0])
    y_test_pred = svr.predict(x_test).reshape(-1, 1)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)
    return test_timestamps, y_test_pred_inv.flatten()


def train_arima(train, test, order_temp, order_humi, steps):
    arima_temp = ARIMA(train["temperature"], order=order_temp).fit()
    arima_humi = ARIMA(train["humidity"], order=order_humi).fit()
    temp_forecast = arima_temp.forecast(steps=steps)
    humi_forecast = arima_humi.forecast(steps=steps)
    df = temp_forecast.reset_index()
    df.columns = ["timestamp", "predict_temp"]
    df["predict_humi"] = humi_forecast.values
    return df


def train_sarima(
    train, test, order_temp, seasonal_temp, order_humi, seasonal_humi, steps
):
    sarima_temp = SARIMAX(
        train["temperature"], order=order_temp, seasonal_order=seasonal_temp
    ).fit(disp=False)
    sarima_humi = SARIMAX(
        train["humidity"], order=order_humi, seasonal_order=seasonal_humi
    ).fit(disp=False)
    temp_forecast = sarima_temp.forecast(steps=steps)
    humi_forecast = sarima_humi.forecast(steps=steps)
    df = temp_forecast.reset_index()
    df.columns = ["timestamp", "predict_temp"]
    df["predict_humi"] = humi_forecast.values
    return df


def build_svr_df(df):
    temp_timestamps, temp_pred = train_svr(
        df, "temperature", SVR_TIMESTEPS, SVR_TEMP_PARAMS
    )
    humi_timestamps, humi_pred = train_svr(
        df, "humidity", SVR_TIMESTEPS, SVR_HUMI_PARAMS
    )
    return pd.DataFrame(
        {
            "timestamp": temp_timestamps,
            "predict_temp": temp_pred,
            "predict_humi": humi_pred,
        }
    )


def to_iso_records(df):
    result = df.reset_index().to_dict(orient="records")
    for record in result:
        if isinstance(record["timestamp"], pd.Timestamp):
            record["timestamp"] = record["timestamp"].isoformat()
    return result


def update_data():
    global latest_df, latest_arima_df, latest_sarima_df, latest_svr_df
    while True:
        try:
            feeds = fetch_data(API_URL)
            df = preprocess_data(feeds)
            train, test = train_test_split(df, test_size=TEST_SIZE, shuffle=False)

            # Lebih ringan supaya nggak timeout
            arima_df = train_arima(
                train,
                test,
                order_temp=(2, 1, 3),
                order_humi=(2, 1, 4),
                steps=len(test) + 72,
            )

            sarima_df = train_sarima(
                train,
                test,
                order_temp=(0, 0, 2),
                seasonal_temp=(1, 1, 2, 24),
                order_humi=(0, 1, 1),
                seasonal_humi=(2, 1, 0, 24),
                steps=len(test) + 72,
            )

            svr_df = build_svr_df(df)

            latest_df = df
            latest_arima_df = arima_df
            latest_sarima_df = sarima_df
            latest_svr_df = svr_df

            print("Data updated at", pd.Timestamp.now())
        except Exception as e:
            print("Error updating data:", e)
        time.sleep(UPDATE_INTERVAL)


# Flask API
app = Flask(__name__)
CORS(app)


def start_background_job():
    threading.Thread(target=update_data, daemon=True).start()


start_background_job()


@app.route("/", methods=["GET"])
def home():
    return """
    <h1>Time Series Forecasting API</h1>
    <ul>
        <li><a href="/resampling">Resampling Data</a></li>
        <li><a href="/arima_forecast">ARIMA Forecast</a></li>
        <li><a href="/sarima_forecast">SARIMA Forecast</a></li>
        <li><a href="/svr_forecast">SVR Forecast</a></li>
        <li><a href="/describe">Describe Data</a></li>
    </ul>
    """


@app.route("/resampling", methods=["GET"])
def resampling_data():
    if latest_df is not None:
        return jsonify(to_iso_records(latest_df))
    return jsonify({"error": "Data not ready"})


@app.route("/arima_forecast", methods=["GET"])
def arima_forecast_data():
    if latest_arima_df is not None:
        return jsonify(to_iso_records(latest_arima_df))
    return jsonify({"error": "Data not ready"})


@app.route("/sarima_forecast", methods=["GET"])
def sarima_forecast_data():
    if latest_sarima_df is not None:
        return jsonify(to_iso_records(latest_sarima_df))
    return jsonify({"error": "Data not ready"})


@app.route("/svr_forecast", methods=["GET"])
def svr_forecast_data():
    if latest_svr_df is not None:
        return jsonify(to_iso_records(latest_svr_df))
    return jsonify({"error": "Data not ready"})


@app.route("/describe", methods=["GET"])
def describe_data():
    if latest_df is not None:
        result = latest_df.describe().to_dict()
        return jsonify(result)
    return jsonify({"error": "Data not ready"})


if __name__ == "__main__":
    # Railway akan set PORT env, default 8080
    import os

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
