# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 05:02:08 2022

@author: perletti
"""

#Load the required libraries
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

# list of crptocurrencies as ticker arguments
straCrypto = ['BTC-USD', 'ETH-USD', 'BNB-USD','CRO-USD']
start = dt.date(2020, 1, 14)
end=dt.datetime.now()
data = yf.download(straCrypto, start,  end )
data.head()


adj_close=data['Adj Close']
adj_close.head()

# ploting the adjusted closing price
fig, handleGui =plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
handleGui[0,0].plot(adj_close['BNB-USD'])
handleGui[0,0].set_title('BNB')
handleGui[0,1].plot(adj_close['BTC-USD'])
handleGui[0,1].set_title('BTC')
handleGui[1,0].plot(adj_close['ETH-USD'])
handleGui[1,0].set_title('ETH')
handleGui[1,1].plot(adj_close['CRO-USD'])
handleGui[1,1].set_title('CRO')
plt.show()

# Returns 
returns = adj_close.pct_change().dropna(axis=0)
#spy the first 5 rows of the data
returns.head()



#plotting the returns
fig, handleGui = plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
handleGui[0,0].plot(returns['BNB-USD'])
handleGui[0,0].set_title('BNB')
handleGui[0,0].set_ylim([-0.5,0.5])
handleGui[0,1].plot(returns['BTC-USD'])
handleGui[0,1].set_title('BTC')
handleGui[0,1].set_ylim([-0.5,0.5])
handleGui[1,0].plot(returns['ETH-USD'])
handleGui[1,0].set_title('ETH')
handleGui[1,0].set_ylim([-0.5,0.5])
handleGui[1,1].plot(returns['CRO-USD'])
handleGui[1,1].set_title('CRO')
handleGui[1,1].set_ylim([-0.5,0.5])
plt.show()



#ploting the histogram
fig, handleGui = plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
handleGui[0,0].hist(returns['BNB-USD'], bins=50, range=(-0.2, 0.2))
handleGui[0,0].set_title('BNB')
handleGui[0,1].hist(returns['BTC-USD'], bins=50, range=(-0.2, 0.2))
handleGui[0,1].set_title('BTCB')
handleGui[1,0].hist(returns['ETH-USD'], bins=50, range=(-0.2, 0.2))
handleGui[1,0].set_title('ETH')
handleGui[1,1].hist(returns['CRO-USD'], bins=50, range=(-0.2, 0.2))
handleGui[1,1].set_title('CRO')
plt.show()


# Cumulative return series
cum_returns = ((1 + returns).cumprod() - 1) *100
cum_returns.head()


#compute the correlations
returns.corr()

#plot the correlations
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
plt.show()



# compute a short-term 20-day moving average
MA20 = adj_close.rolling(20).mean()
# compute a Long-term 50-day moving average
MA50 = adj_close.rolling(100).mean()
# compute a Long-term 100-day moving average
MA100 = adj_close.rolling(100).mean()
# ploting the moving average
fig, handleGui = plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
handleGui[0,0].plot(adj_close['BNB-USD'], label= 'price')
handleGui[0,0].plot(MA20['BNB-USD'], label= 'MA20')
handleGui[0,0].plot(MA100['BNB-USD'], label= 'MA100')
handleGui[0,0].set_title('BNB')
handleGui[0,0].legend()
handleGui[0,1].plot(adj_close['BTC-USD'], label= 'price')
handleGui[0,1].plot(MA20['BTC-USD'], label= 'MA20')
handleGui[0,1].plot(MA100['BTC-USD'], label= 'MA100')
handleGui[0,1].set_title('BTC')
handleGui[0,1].legend()
handleGui[1,0].plot(adj_close['ETH-USD'], label= 'price')
handleGui[1,0].plot(MA20['ETH-USD'], label= 'MA20')
handleGui[1,0].plot(MA100['ETH-USD'], label= 'MA100')
handleGui[1,0].set_title('ETH')
handleGui[1,0].legend()
handleGui[1,1].plot(adj_close['CRO-USD'], label= 'price')
handleGui[1,1].plot(MA20['CRO-USD'], label= 'MA20')
handleGui[1,1].plot(MA100['CRO-USD'], label= 'MA100')
handleGui[1,1].set_title('CRO')
handleGui[1,1].legend()
plt.show()