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

def get_Random_weights(lNumStocks):
    #invent weight array
    weight_P0 = np.array(np.random.randint(1,13,lNumStocks)).astype(float)
    #rebalance weights to sum to 1
    weight_P0 /= np.sum(weight_P0)
    return weight_P0

def get_Log_return(P0, r_Tot):
    lDays = len(P0)
    f64a_range = np.arange(lDays, dtype=float)
    f64a_range = f64a_range / f64a_range[-1]
    P1 = P0.copy()
    strColnames = P1.columns
    for ii in range(len(strColnames)):
        P1[strColnames[ii]] = np.exp (f64a_range * r_Tot[ii]) * P0.iloc[0,ii]
    return P1


# %matplotlib qt
plt.close('all')
currency = "EUR"
metric = "Open"
delta = dt.timedelta(days=300)
startDate = dt.datetime.now() - delta
endDate =  dt.datetime.now()

strStock = [ 
'BTC', 'ETH', 'USDC' , 'CRO', 'BNB'
   # 'CRO' , 'EGLD',
   # 'VVS', 'BIFI','ADA',
   # 'SOL', 'LUNA', 'AVAX' , 'DOT' , 
   # 'CRO' , 'WBTC',  'AVAX' , 
   # 'JUV', 'BONDLY'
  ]

lNumStocks = len(strStock)
strStockFiat = [sub + '-' + currency for sub in strStock]
def get_Historical_Data(strStockFiat):
    strCsvFile = f'PortForlioOpt_{startDate.date()}_{endDate.date()}.csv'
    try:
        P0 = pd.read_csv(strCsvFile, parse_dates=[0], dayfirst=True, index_col = 'Date')
    except OSError:
        P0 = web.DataReader(strStockFiat, "yahoo", startDate, endDate)['Close']
        P0.to_csv(strCsvFile)
    # using dropna() function  
    P0.dropna()
    return P0


P0 = get_Historical_Data(strStockFiat)

#convert daily stock prices into daily log(x) returns
# P0 = np.log2(P0*100)
P0 = P0.tail(36)
lNumEpochs = 711
#Array to hold weight for each stock
m_results = np.zeros( (4+lNumStocks-1,lNumEpochs) )
lDays = ( endDate.date() -  startDate.date() ).days + 1
std_max = 0.1
for ii in range(lNumEpochs):
    P0_weights = get_Random_weights(lNumStocks)
    r_Tot = np.log ( P0.iloc[-1] .div( P0.iloc[0] ) )
    
    r_Return_annual = 100 * (np.exp(r_Tot / lDays * 365) - 1)
    P_log = get_Log_return(P0, r_Tot)
    D_t = P0 / P_log - 1
    std_Dt = D_t.std()

    #calculate portfolio return and volatility(standard deviation)
    ii_return  = np.sum(r_Return_annual * P0_weights)
    ii_std_dev = np.sum(std_Dt * P0_weights)

    #store results in results array
    m_results[0,ii] = ii_return
    m_results[1,ii] = ii_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    v1 = r_Return_annual * (1 - ( (std_Dt * P0_weights)/std_max)**0.2) * P0_weights
    
    m_results[2,ii] = np.sum(v1)
    #iterate through the weight vector and add data to results array
    for jj in range(lNumStocks):
        m_results[jj+3,ii] = P0_weights[jj]
plt.close('all')
strColStocks = P0.columns
vColumns = ['ret','stdev','vwr']
vColumns.extend(strColStocks)
results_frame = pd.DataFrame(m_results.T, columns= vColumns )
#locate position of portfolio with highest VWR Ratio
r_Max_VWR = results_frame.iloc[results_frame['vwr'].idxmax()]
print("\n-----------------MAX\n", r_Max_VWR)
#locate positon of portfolio with minimum standard deviation
r_Min_VWR = results_frame.iloc[results_frame['vwr'].idxmin()]
print("\n-----------------MIN\n", r_Min_VWR)
#locate positon of portfolio with minimum standard deviation
r_Min_std = results_frame.iloc[results_frame['stdev'].idxmin()]
print("\n-----------------MIN std\n", r_Min_std)
#create scatter plot coloured by Sharpe Ratio
fig1, ax0 = plt.subplots()
ax0 = plt.scatter(results_frame.stdev,results_frame.ret,c = results_frame.vwr,cmap='RdYlBu')
plt.xlabel('std dev')
plt.ylabel('Returns')
plt.grid(True, color = '#555555')
plt.colorbar()
plt.show()
# weights = weights*100
# print ("Weights to invest in ", stocks , "respectively are")
# print (weights)

strColnames = P_log.columns
for strTitle in strColnames:
    fig2, ax1 = plt.subplots()
    line1, = ax1.plot(P_log.index, P_log[strTitle])
    ax1.plot(P0.index, P0[strTitle])
    # ax1.set_strTitle(ticker, color='white')
    ax1.grid(True, color = '#555555')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    plt.xticks(rotation=60)
    ax1.tick_params(axis = 'x', colors='white')
    ax1.tick_params(axis = 'y', colors='white')
    ax1.set_title(strTitle, color='white')

#calculate mean daily return and covariance of daily returns
# var_matrix = P0_return.cov()*252
# #computing P0 variance
# P0_variance = np.transpose(P0_weights())@var_matrix@P0_weights
# #Computing portfolilo volatility(RISK)
# P0_volatility = np.sqrt(P0_variance)

# print('P0 Variance: ', P0_variance)
# print('P0 Volatility: ', P0_volatility)