import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def fetch_data(symbol, api_key='9ZIZYPIA7MBM5WPJ'):
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}")
    response.raise_for_status()
    data = response.json()
    return data

def preprocess_data(data):
    df = pd.DataFrame.from_dict(data.get('Time Series (Daily)', {}), orient='index')
    
    if df.empty:
        raise ValueError("DataFrame is empty. Check the data retrieval process.")
    
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    df.sort_index(inplace=True)
    
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(all_dates)
    
    df.interpolate(method='linear', inplace=True)
    
    return df

#CODE FOR DAAT VISUALIZATION FROM SEPERATE FILE IN THE FOLDER

"""import requests
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

plt.figure(figsize=(12,6))
plt.plot(df.index,df['close'], label='closing price',color='blue')
plt.title("closing price")
plt.xlabel("date")
plt.ylabel("price")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sma
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['SMA_20'], label='20-Day SMA', linestyle='--', color='red')
plt.title('Stock Closing Prices with 20-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import plotly.graph_objects as go
import plotly.io as pio


trace_close = go.Scatter(x=df.index,y=df['close'],mode='lines',name='closing price',line=dict(color='blue'))
trace_sma = go.Scatter(x=df.index,y=df['SMA_20'],mode='lines',name='20-Day SMA',line=dict(color='red'))
layout = go.Layout(title=f"{symbol} Closing Prices and 20-Day SMA",xaxis=dict(title='Date'),yaxis=dict(title='Price'),hovermode='closest')
fig = go.Figure(data=[trace_close, trace_sma], layout=layout)
pio.show(fig)"""


# CODE FOR APPLYING MODEL FOR IBM DATA FROM OTHER FILE 

"""
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
print(f'RMSE: {rmse}')"""

#RECOMMENDATION MODEL FOR 20 COMAPNIES USING ARIMA MODEL

companies = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Amazon': 'AMZN',
    'Google': 'GOOGL',
    'Facebook': 'META',
    'Tesla': 'TSLA',
    'IBM': 'IBM',
    'NVIDIA': 'NVDA',
    'Netflix': 'NFLX',
    'Adobe': 'ADBE',
    'Intel': 'INTC',
    'Cisco': 'CSCO',
    'Oracle': 'ORCL',
    'Salesforce': 'CRM',
    'Pfizer': 'PFE',
    'Johnson & Johnson': 'JNJ',
    'Procter & Gamble': 'PG',
    'Coca-Cola': 'KO',
    'PepsiCo': 'PEP',
    'McDonald\'s': 'MCD'
}



def forecast_stock_price(df, steps=1):
    series = df['close']
    
    model = ARIMA(series, order=(5, 1, 0))  # Adjust (p, d, q) as needed
    model_fit = model.fit()  # No steps argument here

    forecast = model_fit.forecast(steps=steps)  # Forecast future steps

    return forecast[0]  # Return the first value in the forecast

results = []

for company, symbol in companies.items():
    try:
        data = fetch_data(symbol)
        df = preprocess_data(data)
        
        if not df.empty:  # Check if DataFrame is not empty
            forecast = forecast_stock_price(df)
            current_price = df['close'].iloc[-1]
            recommendation = 'Buy' if forecast > current_price else 'Sell'
            
            results.append({
                'Company': company,
                'Symbol': symbol,
                'Forecast': forecast,
                'Current Price': current_price,
                'Recommendation': recommendation
            })
    except Exception as e:
        print(f"Error processing {company}: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv('recommendations.csv', index=False)

print(results_df)

