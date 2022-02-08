# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 06:49:45 2021

@author: perletti
"""
import pandas as pd
import pandas_datareader as web
#import yfinance as yf
import matplotlib.pyplot as plt
#import mplfinance as mpf 
import seaborn as sns
import datetime as dt

#handle csv files
import numpy as np
#used for scientific calculation
from sklearn.svm import SVR
#used for ml
import matplotlib.pyplot as matplt


# %matplotlib qt
plt.close('all')
currency = "USD"
metric = "Open"

start = dt.date(2022,1, 1)
end =  dt.datetime.now()


dataBtc = web.DataReader('BTC-USD', "yahoo", start, end)


crypto = ['JUV', 'BONDLY'] #'ETH', 'BNB' , 'ADA', 'SOL', 'LUNA', 'DOT',  'MATIC', 'CRO', 'ATOM', 'EGLD' ]
colnames = []

first = True

for ticker in crypto:
    data = web.DataReader(f'{ticker}-{currency}', "yahoo", start, end)
    #data2 =  yf.download('BTC',start,interval="1h")
    if 0:
        data = data.resample('1W').agg({'Open': 'first', 
                                      'High': 'max', 
                                      'Low': 'min', 
                                      'Adj Close': 'last'})
    
    # Divide the DataFrame1 elements by the elements of DataFrame2
    if 0:
        data['Open']      = data['Open'].div(dataBtc['Open']);
        data['High']      = data['High'].div(dataBtc['High']);
        data['Low']       = data['Low'].div(dataBtc['Low']);
        data['Adj Close'] = data['Adj Close'].div(dataBtc['Adj Close']);
    
    
    
    # if first:
    #     combined = data[[metric]].copy()
    #     colnames.append(ticker)
    #     combined.columns = colnames
    #     first = False
    # else:
    #     combined = combined.join(data[metric])
    #     colnames.append(ticker)
    #     combined.columns = colnames

    # plt.yscale('log')
    # for ticker in crypto:
    #     plt.plot(combined[ticker], label = ticker)
                                     
    # plt.legend(loc = 'upper right')   
    # plt.show()                              
                                     
    
    # combined = combined.pct_change().corr(method = "pearson")
    # sns.heatmap(combined, annot = True, cmap = "coolwarm")
    
    
    delta = data['Adj Close'].diff(1)
    delta.dropna(inplace = True) # not a number values
    positive = delta.copy()
    negative = delta.copy()
    
    positive  [positive  < 0] = 0
    negative [negative > 0] = 0
    
    
    # RSI calculation
    def calculate_RSI(positive, negative, days=14):
            
        average_gain = positive.rolling(window=days).mean()
        average_loss = abs ( negative.rolling(window=days).mean() )
        relative_strength = average_gain / average_loss
        RSI = 100.0 / (1.0 + relative_strength)
        return RSI
    
    RSI_14 = calculate_RSI(positive, negative, 14)
    # RSI_21 = calculate_RSI(positive, negative, 21)
    
    # RSI calculation
    
    lDayK = 14
    lDayD = 3 # mobile average
    v14Low  = data['Low'].rolling(lDayK).min()
    v14High = data['High'].rolling(lDayK).max()
    vK = ( data['Adj Close'] - v14Low ) * 100.0 / ( v14High - v14Low ) 
    vD = vK.rolling(lDayD).mean()
    
   
    
    sEMA = (26,12,9)
    longEMA = data['Adj Close'].ewm(span=sEMA[0], adjust=False, min_periods=sEMA[0]).mean()
    shortEMA = data['Adj Close'].ewm(span=sEMA[1], adjust=False, min_periods=sEMA[1]).mean()
    vMACD = (  shortEMA - longEMA )
    vSignal = vMACD.ewm(span=sEMA[2], adjust=False,min_periods=9).mean()
    vMacdDiff = ( vMACD - vSignal )
       

    
    combined = pd.DataFrame() # combined data frame
    
    combined['Adj Close'] = data ['Adj Close']
    combined['RSI_14'] = (100.0-RSI_14)
    combined['%K'] = vK
    combined['%D'] = vD
    plt.figure(figsize = (12,8))
    #figA.canvas.mpl_connect('pick_event', DataCursor(plt.gca()))
    
    ax1 = plt.subplot(311)
    line1, = ax1.plot(combined.index, combined['Adj Close'], color = 'lightgray')
    
    ax1.set_title(ticker, color='white')
    ax1.grid(True, color = '#555555')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis = 'x', colors='white')
    ax1.tick_params(axis = 'y', colors='white')
    
    
    ax2 = plt.subplot(312, sharex = ax1)
    # line2, = ax2.plot(combined.index, combined['RSI_14'], color = 'lightgray')
    ax2.plot(combined.index, combined['%K'], color = 'magenta')
    ax2.plot(combined.index, combined['%D'], color = 'cyan')
    
    ax2.axhline(0, linestyle='--', alpha =0.5, color = '#FF0000')
    ax2.axhline(10,linestyle='--', alpha =0.5, color = '#FFaa00')
    ax2.axhline(20,linestyle='--', alpha =0.5, color = '#00FF00')
    ax2.axhline(30,linestyle='--', alpha =0.5, color = '#CCccCC')
    
    ax2.axhline(100,linestyle='--', alpha =0.5, color = '#FF0000')
    ax2.axhline(90 ,linestyle='--', alpha =0.5, color = '#FFaa00')
    ax2.axhline(80 ,linestyle='--', alpha =0.5, color = '#00FF00')
    ax2.axhline(70 ,linestyle='--', alpha =0.5, color = '#CCccCC')
    
    
    ax2.set_title("RSI", color='white')
    ax2.grid(False)
    ax2.set_axisbelow(False)
    ax2.set_facecolor('black')
    ax2.tick_params(axis = 'x', colors='white')
    ax2.tick_params(axis = 'y', colors='white')
    
    
    ax3 = plt.subplot(313, sharex = ax1)
    ax3.plot(data.index, vMACD, label =  'MACD', color = 'green')
    ax3.plot(data.index, vSignal, label =  'Signal', color = 'red')
    ax3.plot(data.index, vMacdDiff, label =  'Delta', color = 'grey')
    ax3.axhline(0,linestyle='--', alpha =0.5, color = '#CCccCC')
    ax3.grid(True, color = '#555555')
    ax3.set_axisbelow(True)
    ax3.set_facecolor('black')
    ax3.figure.set_facecolor('#121212')
    ax3.tick_params(axis = 'x', colors='white')
    ax3.tick_params(axis = 'y', colors='white')
    
    





















