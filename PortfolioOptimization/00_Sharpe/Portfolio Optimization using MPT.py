#https://www.analyticsvidhya.com/blog/2021/04/portfolio-optimization-using-mpt-in-python/
import ssl
from functools import wraps
def sslwrap(func):
    @wraps(func)
    def bar(*args, **kw):
        kw['ssl_version'] = ssl.PROTOCOL_TLSv1
        return func(*args, **kw)
    return bar
ssl.wrap_socket = sslwrap(ssl.wrap_socket)
#Technique used to determine weights: Sharpe ratio
#Data source: Yahoo finance
#Monte Carlo simulations to assign random weights to the stocks and calculate volatility
#Start dateeis taken as Jan 1,2016

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

#list of stocks in portfolio
stocks = ['BTC-USD', 'ETH-USD' ,'BNB-USD', 'ADA-USD', 'SOL-USD', 'CRO-USD', 'DOT-USD']
# stocks = [ 'LYYB.F', 'UIMR.AS', 'INTC', 'MDT' , 'PFE','ENGI.PA']
lLenStocks = len(stocks)
#requests.post(url, data=payload, headers=headers, verify=False) 
#download daily price data for each of the stocks in the portfolio
portfolio = web.DataReader(stocks,data_source='yahoo',start='1/2/2020', end = '2/22/2022')['Adj Close']
# using dropna() function  
portfolio.dropna()
#convert daily stock prices into daily returns
returns_portfolio = portfolio.pct_change().apply(lambda x: np.log(1+x))
#invent weight array
weight_portfolio = np.array(np.random.randint(1,10,lLenStocks)).astype(float)
    #rebalance weights to sum to 1
weight_portfolio /= np.sum(weight_portfolio)

#Dot product to wiegth investments
portfolio_return = weight_portfolio.dot(weight_portfolio)



 
#calculate mean daily return and covariance of daily returns
var_matrix = returns_portfolio.cov()*252
#computing portfolio variance
portfolio_variance = np.transpose(weight_portfolio)@var_matrix@weight_portfolio
#Computing portfolilo volatility(RISK)
portfolio_volatility = np.sqrt(portfolio_variance)

print('Portfolio Variance: ', portfolio_variance)
print('Portfolio Volatility: ', portfolio_volatility)
#init empty list storing returns
port_returns = []
port_volatility = []
port_weights = []

num_assets = len(portfolio.columns)
num_portfolios = 710

#computing indiviual assets return
individual_rets = portfolio.resample('Y').last().pct_change().mean()

for port in range(num_portfolios):
    #invent weight array
    weights = np.random.random(num_assets)    #rebalance weights to sum to 1
    weights /= np.sum(weights)
    port_weights.append(weights)
    #Returns = dotProd indiviual expected return * weights
    returns = np.dot(weights, individual_rets)
    port_returns.append(returns)
    
    #computing portfolio variance
    var = var_matrix.mul(weights , axis = 0).mul(weights, axis=1).sum().sum()
    #daily std: volatolity is square root of variance
    sd = np.sqrt(var)
    ann_sd = sd*np.sqrt(252)
    port_volatility.append(ann_sd)
    

#Now create a Dictionary of returns and volatility
data = {'Returns': port_returns, 'Volatility': port_volatility}

for counter, symbol in enumerate(portfolio.columns.tolist()):
    data[symbol + ' weight'] = [w[counter] for w in port_weights]

#converting dictionary to dataframe
portfolios_V1 = pd.DataFrame(data)
portfolios_V1.dropna()



min_vol_port = portfolios_V1.iloc[portfolios_V1['Volatility'].idxmin()]

print(min_vol_port)

fig2, ax1 = plt.subplots()
ax1.grid(True, color = '#555555')
ax1.set_axisbelow(True)
ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')
plt.xticks(rotation=60)
ax1.tick_params(axis = 'x', colors='white')
ax1.tick_params(axis = 'y', colors='white')
#plot efficent frontier
ax1.scatter(portfolios_V1['Volatility'], portfolios_V1['Returns'], marker = 'o', color='c',
                           s=15, alpha=0.8)
rf = 0.01
optimal_risk_port = portfolios_V1.iloc[   ((portfolios_V1['Returns']-rf)/portfolios_V1['Volatility']).idxmax() ]          


ax1.scatter(min_vol_port[1], min_vol_port[0],color='r', marker='h', s=500)
ax1.scatter(optimal_risk_port[1], optimal_risk_port[0], color='b', marker='*', s=500)












