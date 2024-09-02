import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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



print(df.head())
print(df.columns)  


#plot of closing price

"""plt.figure(figsize=(12,6))
plt.plot(df.index,df['close'], label='closing price',color='blue')
plt.title("closing price")
plt.xlabel("date")
plt.ylabel("price")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()"""

"""sma
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['SMA_20'], label='20-Day SMA', linestyle='--', color='red')
plt.title('Stock Closing Prices with 20-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()"""

import plotly.graph_objects as go
import plotly.io as pio


trace_close = go.Scatter(x=df.index,y=df['close'],mode='lines',name='closing price',line=dict(color='blue'))
trace_sma = go.Scatter(x=df.index,y=df['SMA_20'],mode='lines',name='20-Day SMA',line=dict(color='red'))
layout = go.Layout(title=f"{symbol} Closing Prices and 20-Day SMA",xaxis=dict(title='Date'),yaxis=dict(title='Price'),hovermode='closest')
fig = go.Figure(data=[trace_close, trace_sma], layout=layout)
pio.show(fig)