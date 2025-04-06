import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configure the Streamlit page
st.set_page_config(page_title="Tech Stock Forecast Dashboard", layout="wide")

# Sidebar: User configuration
st.sidebar.header("Configuration")
stock_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL")
forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=1, max_value=30, value=7)
train_years = st.sidebar.slider("Training Period (years)", min_value=1, max_value=10, value=3)
alpha_api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
update_data = st.sidebar.button("Update Data & Retrain")

st.title("📈 Real-Time Tech Stock Price Forecast")
st.markdown(f"**Stock:** {stock_symbol} &nbsp;&nbsp; **Forecast Horizon:** {forecast_days} day(s)")

if not alpha_api_key:
    st.error("Please enter your Alpha Vantage API key in the sidebar.")
    st.stop()

# 1. Data Ingestion using Alpha Vantage
end_date = datetime.now()
start_date = end_date - timedelta(days=train_years * 365)

st.write("Fetching data from Alpha Vantage...")
try:
    ts = TimeSeries(key=alpha_api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')
except Exception as e:
    st.error(f"Error fetching data for {stock_symbol}: {e}")
    st.stop()

# Process data: use '4. close' column and convert index to datetime
data = data.rename(columns={"4. close": "Close"})
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Use the 'Close' price for forecasting
df = data[["Close"]].dropna()

if df.empty:
    st.error("No data found. Please try a different stock ticker.")
    st.stop()

# Display historical prices
st.subheader("Historical Price")
st.line_chart(df["Close"], height=250)

# 2. Hybrid Model Training

# 2.a ARIMA Model for baseline trend
arima_train = df["Close"].values
use_arima = True
if len(arima_train) < max(30, 2 * forecast_days):
    use_arima = False

if use_arima:
    try:
        # Fit an ARIMA model with order (2, 1, 1)
        arima_model = ARIMA(arima_train, order=(2, 1, 1))
        arima_res = arima_model.fit()
    except Exception as e:
        st.error(f"ARIMA model fitting failed: {e}")
        use_arima = False
        arima_res = None
else:
    arima_res = None

# 2.b Compute residuals for LSTM training
if use_arima and arima_res is not None:
    # In-sample ARIMA predictions
    arima_fitted = arima_res.predict()
    residuals = arima_train - arima_fitted
else:
    residuals = arima_train

# 2.c Prepare LSTM training data from residuals
window_size = 60  # lookback window
if len(residuals) < window_size + 1:
    window_size = max(1, len(residuals) - 1)
series = residuals.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

X_train, y_train = [], []
for i in range(window_size, len(series_scaled)):
    X_train.append(series_scaled[i - window_size:i, 0])
    y_train.append(series_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 2.d LSTM Model Training (if sufficient data)
lstm_model = None
if X_train.shape[0] >= 1:
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', return_sequences=False, input_shape=(window_size, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

# 3. Forecasting

# ARIMA forecast for baseline
last_close = df["Close"].iloc[-1]
if use_arima and arima_res is not None:
    try:
        arima_forecast = arima_res.forecast(steps=forecast_days)
    except Exception as e:
        st.error(f"ARIMA forecasting failed: {e}")
        arima_forecast = np.array([last_close] * forecast_days)
else:
    arima_forecast = np.array([last_close] * forecast_days)

# LSTM forecast for residuals (iterative one-step prediction)
lstm_forecast = np.zeros(forecast_days)
if lstm_model:
    recent_series = series[-window_size:]
    recent_scaled = scaler.transform(recent_series.reshape(-1, 1))
    for i in range(forecast_days):
        x_input = recent_scaled[-window_size:].reshape(1, window_size, 1)
        pred_scaled = lstm_model.predict(x_input, verbose=0)
        pred_residual = scaler.inverse_transform(pred_scaled)[0, 0]
        lstm_forecast[i] = pred_residual
        new_val_scaled = pred_scaled[0, 0]
        recent_scaled = np.append(recent_scaled, [[new_val_scaled]], axis=0)

# Combine ARIMA and LSTM forecasts
combined_forecast = arima_forecast + lstm_forecast

# 4. Visualization: Forecast vs. Actual
last_date = df.index[-1]
forecast_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
forecast_df = pd.DataFrame({
    'Forecast': combined_forecast,
    'Baseline': arima_forecast,
}, index=pd.to_datetime(forecast_dates))

display_df = pd.concat([df.tail(60)[["Close"]], forecast_df])

st.subheader("Forecast vs. Actual")
st.line_chart(display_df)

st.markdown(f"**Latest Actual Price:** ${last_close:.2f}")
st.markdown(f"**Forecasted Price ({forecast_days} days ahead):** ${combined_forecast[-1]:.2f}")

# Matplotlib visualization for a more customized chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df.index[-200:], df['Close'][-200:], label="Historical Close")
ax.plot(forecast_df.index, forecast_df["Forecast"], label="Forecasted Close", marker='o')
ax.set_title(f"{stock_symbol} Stock Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
st.pyplot(fig)
