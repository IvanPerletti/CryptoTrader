# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:03:04 2023
https://github.com/LastAncientOne/Stock_Analysis_For_Quant/blob/master/Python_Stock/Stock_Alpha_Beta.ipynb
@author: perletti
"""

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import dateutil.relativedelta
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf

# input
ticker =     'MSFT' #Microsoft **
spx = "^GSPC"
end =  dt.datetime.today().date()
start = end - dateutil.relativedelta.relativedelta(years=1)


# Read data 
stock = yf.download(ticker,start,end)
market = yf.download(spx, start, end)

# View columns 
stock.head()

# View columns 
market.head()

prices = stock['Adj Close']
values = market['Adj Close']

#ret = prices.pct_change(1)[1:]
#ret = np.log(prices/prices.shift(1))
ret = (np.log(prices) - np.log(prices.shift(1))).dropna()


mrk = values.pct_change(1).dropna()
mrk.head()

from scipy import stats

beta, alpha, r_value, p_value, std_err = stats.linregress(ret, mrk)


print("Beta: 			%9.6f" % beta)
print("Alpha: 			%9.6f" % alpha)
print("R-Squared: 		%9.6f" % r_value)
print("p-value: 		%9.6f" % p_value)
print("Standard Error: 	%9.6f" % std_err)