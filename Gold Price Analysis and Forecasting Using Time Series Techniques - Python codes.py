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


# 1. SUMMARY STATISTICS

print("\nSUMMARY STATISTICS")

mean_price = df['Close'].mean()
median_price = df['Close'].median()
min_price = df['Close'].min()
max_price = df['Close'].max()
std_price = df['Close'].std()
skewness = df['Close'].skew()
kurtosis = df['Close'].kurt()

summary_table = pd.DataFrame({
    'Metric': [
        'Mean Price',
        'Median Price',
        'Minimum Price',
        'Maximum Price',
        'Standard Deviation',
        'Skewness',
        'Kurtosis'
    ],
    'Value': [
        round(mean_price,2),
        round(median_price,2),
        round(min_price,2),
        round(max_price,2),
        round(std_price,2),
        round(skewness,4),
        round(kurtosis,4)
    ]
})

print(summary_table)


# 2. ADF TEST RESULTS

from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Close'])

adf_table = pd.DataFrame({
    'Metric': [
        'ADF Statistic',
        'p-value',
        'Critical Value (1%)',
        'Critical Value (5%)',
        'Critical Value (10%)'
    ],
    'Value': [
        round(result[0],4),
        round(result[1],6),
        round(result[4]['1%'],4),
        round(result[4]['5%'],4),
        round(result[4]['10%'],4)
    ]
})

print("\nADF TEST RESULTS")
print(adf_table)


# 3. TRAIN TEST SPLIT

train_size = int(len(df) * 0.80)

train = df['Close'][:train_size]
test = df['Close'][train_size:]


# 4. ARIMA MODEL

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

predictions = model_fit.forecast(
    steps=len(test)
)


# 5. FORECAST EVALUATION METRICS

rmse = np.sqrt(
    mean_squared_error(
        test,
        predictions
    )
)

mae = mean_absolute_error(
    test,
    predictions
)

mape = np.mean(
    np.abs(
        (test - predictions) / test
    )
) * 100

metrics_table = pd.DataFrame({
    'Metric': [
        'RMSE',
        'MAE',
        'MAPE'
    ],
    'Value': [
        round(rmse,4),
        round(mae,4),
        round(mape,4)
    ]
})

print("\nFORECAST EVALUATION METRICS")
print(metrics_table)


# 6. 30-DAY FORECAST WITH 95% CONFIDENCE INTERVAL

final_model = ARIMA(
    df['Close'],
    order=(5,1,0)
)

final_fit = final_model.fit()

forecast_result = final_fit.get_forecast(
    steps=30
)

forecast = forecast_result.predicted_mean

confidence_intervals = (
    forecast_result.conf_int()
)

forecast_table = pd.DataFrame({
    'Forecast Price': forecast,
    'Lower 95% CI':
        confidence_intervals.iloc[:,0],
    'Upper 95% CI':
        confidence_intervals.iloc[:,1]
})

print("\n30-DAY FORECAST WITH CONFIDENCE INTERVALS")
print(forecast_table)



# 7. PRICE TREND

plt.figure(figsize=(12,5))
plt.plot(df['Close'])
plt.title("Gold Price Trend")
plt.show()


# 8. DAILY RETURNS

plt.figure(figsize=(12,5))
plt.plot(df['Return'])
plt.title("Daily Returns")
plt.show()


# 9. VOLATILITY

plt.figure(figsize=(12,5))
plt.plot(df['Volatility'])
plt.title("30-Day Rolling Volatility")
plt.show()


# 10. MOVING AVERAGES

plt.figure(figsize=(12,5))
plt.plot(df['Close'], label="Close")
plt.plot(df['MA50'], label="MA50")
plt.plot(df['MA200'], label="MA200")
plt.legend()
plt.title("Moving Averages")
plt.show()


# 11. CORRELATION HEATMAP

plt.figure(figsize=(6,5))
sns.heatmap(df[['Open','High','Low','Close','Volume']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# 12. ARIMA FORECAST

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