# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:38:24 2022
https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
@author: perletti
"""
import pandas as pd
import pandas_datareader as web
#import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
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
currency = "EUR"
metric = "Open"
delta = dt.timedelta(days=9)
startDate = dt.datetime.now() - delta
endDate =  dt.datetime.now()

strStock = [ 
'BTC', 'ETH', 'BNB' ,
   # 'CRO' , 'EGLD',
   # 'VVS', 'BIFI','ADA',
   # 'SOL', 'LUNA', 'AVAX' , 'DOT' , 
   # 'CRO' , 'WBTC',  'AVAX' , 
   # 'JUV', 'BONDLY'
  ]

lNumStocks = len(strStock)
strStockFiat = [sub + '-' + currency for sub in strStock]

strCsvFile = f'PortForlioOpt_{startDate.date()}_{endDate.date()}.csv'
try:
    portfolio = pd.read_csv(strCsvFile, parse_dates=[0], dayfirst=True, index_col = 'Date')
except OSError:
    portfolio = web.DataReader(strStockFiat, "yahoo", startDate, endDate)['Close']
    portfolio.to_csv(strCsvFile)


# using dropna() function  
portfolio.dropna()
#convert daily stock prices into daily returns
returns_portfolio = portfolio.pct_change().apply(lambda x: np.log(1+x))
#invent weight array
weight_portfolio = np.array(np.random.randint(1,10,lNumStocks)).astype(float)
    #rebalance weights to sum to 1
weight_portfolio /= np.sum(weight_portfolio)

#Dot product to wiegth investments
lDays = ( endDate.date() -  startDate.date() ).days + 1
portfolio_return = weight_portfolio.dot(weight_portfolio)
r_Tot = np.log ( portfolio.iloc[-1] .div( portfolio.iloc[0] ) )
r_Bar = r_Tot / ( lDays )
r_Norm = 100 * (np.exp(r_Bar * 365) - 1)

def get_Log_return(portfolio, r_Bar):
    lDays = len(portfolio)
    P0 = pd.concat( [portfolio.iloc[0]]*lDays , 1).T
    f64a_range = np.arange(lDays, dtype=float)
    f64a_range = f64a_range / f64a_range[-1]
    f64a_range = np.exp(f64a_range * r_Tot[1])
    # P1 = pd.DataFrame()
    P1 = P0.iloc[:, 1].multiply(f64a_range, axis=0)
    return P1

P_log = get_Log_return(portfolio, r_Bar)
D_t = portfolio / P_log -1

    
# zeroVarPrice = portfolio.iloc[0].dot(np.exp(r_Bar*range(lDays)))
# diffPrice_ratio =  portfolio.div(zeroVarPrice)

#calculate mean daily return and covariance of daily returns
var_matrix = returns_portfolio.cov()*252
#computing portfolio variance
portfolio_variance = np.transpose(weight_portfolio)@var_matrix@weight_portfolio
#Computing portfolilo volatility(RISK)
portfolio_volatility = np.sqrt(portfolio_variance)

print('Portfolio Variance: ', portfolio_variance)
print('Portfolio Volatility: ', portfolio_volatility)