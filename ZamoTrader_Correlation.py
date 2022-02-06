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
cryptocurrencies = ['BTC-USD', 'ETH-USD', 'BNB-USD','XRP-USD']
start = dt.date(2020, 1, 14)
end=dt.datetime.now()
data = yf.download(cryptocurrencies, start,
                end )
data.head()


adj_close=data['Adj Close']
adj_close.head()

# ploting the adjusted closing price
fig, axs =plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
axs[0,0].plot(adj_close['BNB-USD'])
axs[0,0].set_title('BNB')
axs[0,1].plot(adj_close['BTC-USD'])
axs[0,1].set_title('BTC')
axs[1,0].plot(adj_close['ETH-USD'])
axs[1,0].set_title('ETH')
axs[1,1].plot(adj_close['XRP-USD'])
axs[1,1].set_title('XRP')
plt.show()

# Returns i.e. percentage change in the adjusted close price and drop the first row with NA's
returns = adj_close.pct_change().dropna(axis=0)
#view the first 5 rows of the data frame
returns.head()



#ploting the returns
fig, axs = plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
axs[0,0].plot(returns['BNB-USD'])
axs[0,0].set_title('BNB')
axs[0,0].set_ylim([-0.5,0.5])
axs[0,1].plot(returns['BTC-USD'])
axs[0,1].set_title('BTC')
axs[0,1].set_ylim([-0.5,0.5])
axs[1,0].plot(returns['ETH-USD'])
axs[1,0].set_title('ETH')
axs[1,0].set_ylim([-0.5,0.5])
axs[1,1].plot(returns['XRP-USD'])
axs[1,1].set_title('XRP')
axs[1,1].set_ylim([-0.5,0.5])
plt.show()



#ploting the histogram
fig, axs = plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
axs[0,0].hist(returns['BNB-USD'], bins=50, range=(-0.2, 0.2))
axs[0,0].set_title('BNB')
axs[0,1].hist(returns['BTC-USD'], bins=50, range=(-0.2, 0.2))
axs[0,1].set_title('BTCB')
axs[1,0].hist(returns['ETH-USD'], bins=50, range=(-0.2, 0.2))
axs[1,0].set_title('ETH')
axs[1,1].hist(returns['XRP-USD'], bins=50, range=(-0.2, 0.2))
axs[1,1].set_title('XRP')
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
fig, axs = plt.subplots(2,2,figsize=(16,8),gridspec_kw ={'hspace': 0.2, 'wspace': 0.1})
axs[0,0].plot(adj_close['BNB-USD'], label= 'price')
axs[0,0].plot(MA20['BNB-USD'], label= 'MA20')
axs[0,0].plot(MA100['BNB-USD'], label= 'MA100')
axs[0,0].set_title('BNB')
axs[0,0].legend()
axs[0,1].plot(adj_close['BTC-USD'], label= 'price')
axs[0,1].plot(MA20['BTC-USD'], label= 'MA20')
axs[0,1].plot(MA100['BTC-USD'], label= 'MA100')
axs[0,1].set_title('BTC')
axs[0,1].legend()
axs[1,0].plot(adj_close['ETH-USD'], label= 'price')
axs[1,0].plot(MA20['ETH-USD'], label= 'MA20')
axs[1,0].plot(MA100['ETH-USD'], label= 'MA100')
axs[1,0].set_title('ETH')
axs[1,0].legend()
axs[1,1].plot(adj_close['XRP-USD'], label= 'price')
axs[1,1].plot(MA20['XRP-USD'], label= 'MA20')
axs[1,1].plot(MA100['XRP-USD'], label= 'MA100')
axs[1,1].set_title('XRP')
axs[1,1].legend()
plt.show()