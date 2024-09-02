import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def fetch_data(symbol, interval='5min',api_key='9ZIZYPIA7MBM5WPJ'):
    response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo")
    response.raise_for_status()
    data = response.json()
    return data

symbol = 'IBM'
data=fetch_data(symbol)

time_series = data.get('Time Series (Daily)', {})

df = pd.DataFrame.from_dict(time_series, orient='index')

#print(df.head())
#print(df.columns)

df.index = pd.to_datetime(df.index)
df = df.astype(float)

df.columns = ['open', 'high', 'low', 'close', 'volume']

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['open', 'high', 'low', 'close', 'volume']] = imputer.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

from sklearn.preprocessing import   MinMaxScaler
scaler = MinMaxScaler()
df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
 
df['SMA_20'] = df['close'].rolling(window=20).mean()



"""print(df.head())
print(df.columns)"""  

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
df.index = pd.to_datetime(df.index)
ts = df['close']

train_size = int(len(ts)*0.8)
train, test =ts[:train_size], ts[train_size:]

model = sm.tsa.ARIMA(train, order=(5,1,0))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=len(test))
print(forecast)

from sklearn.metrics import mean_squared_error, mean_absolute_error


mse = mean_squared_error(test, forecast)
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mse)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
