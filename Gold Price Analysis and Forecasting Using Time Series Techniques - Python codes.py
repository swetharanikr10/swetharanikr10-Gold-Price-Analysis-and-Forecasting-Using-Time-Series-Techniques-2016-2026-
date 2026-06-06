# Gold Price Analysis and Forecasting Using Time Series Techniques

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


# LOAD DATA

df = pd.read_csv("Data/gold_historical_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)


# RETURNS + VOLATILITY

df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(30).std()


# MOVING AVERAGES

df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()


# 1. PRICE TREND

plt.figure(figsize=(12,5))
plt.plot(df['Close'])
plt.title("Gold Price Trend")
plt.show()


# 2. DAILY RETURNS

plt.figure(figsize=(12,5))
plt.plot(df['Return'])
plt.title("Daily Returns")
plt.show()


# 3. VOLATILITY

plt.figure(figsize=(12,5))
plt.plot(df['Volatility'])
plt.title("30-Day Rolling Volatility")
plt.show()


# 4. MOVING AVERAGES

plt.figure(figsize=(12,5))
plt.plot(df['Close'], label="Close")
plt.plot(df['MA50'], label="MA50")
plt.plot(df['MA200'], label="MA200")
plt.legend()
plt.title("Moving Averages")
plt.show()


# 5. CORRELATION HEATMAP

plt.figure(figsize=(6,5))
sns.heatmap(df[['Open','High','Low','Close','Volume']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# 6. ARIMA FORECAST

model = ARIMA(df['Close'], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=30)

future_dates = pd.date_range(df.index[-1], periods=31, freq='D')[1:]

plt.figure(figsize=(12,5))
plt.plot(df['Close'], label="Historical")
plt.plot(future_dates, forecast, label="Forecast")
plt.legend()
plt.title("ARIMA Forecast (30 Days)")
plt.show()