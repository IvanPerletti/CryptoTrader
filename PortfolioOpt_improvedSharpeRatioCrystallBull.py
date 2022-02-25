# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:38:24 2022
https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
@author: perletti
"""
import pandas as pd
import pandas_datareader as web
from sklearn import datasets, linear_model
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

def dt64_to_float(dt64):
    """Converts numpy.datetime64 to year as float.

    Rounded to days

    Parameters
    ----------
    dt64 : np.datetime64 or np.ndarray(dtype='datetime64[X]')
        date data

    Returns
    -------
    float or np.ndarray(dtype=float)
        Year in floating point representation
    """

    year = dt64.astype('M8[Y]')
    # print('year:', year)
    days = (dt64 - year).astype('timedelta64[D]')
    # print('days:', days)
    year_next = year + np.timedelta64(1, 'Y')
    # print('year_next:', year_next)
    days_of_year = (year_next.astype('M8[D]') - year.astype('M8[D]')
                    ).astype('timedelta64[D]')
    # print('days_of_year:', days_of_year)
    dt_float = 1970 + year.astype(float) + days / (days_of_year)
    # print('dt_float:', dt_float)
    return dt_float
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
def get_Random_weights(lNumStocks):
    #invent weight array
    weight_P0 = np.array(np.random.randint(1,13,lNumStocks)).astype(float)
    #rebalance weights to sum to 1
    weight_P0 = weight_P0**2
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

def get_Line_return(P0, r_Tot):
    lDays = len(P0)
    f64a_range = np.arange(lDays, dtype=float)
    f64a_range = f64a_range / f64a_range[-1]
    P1 = P0.copy()
    strColnames = P1.columns
    for ii in range(len(strColnames)):
        x=np.array(P0.index.values, dtype=float)
        y=np.array(P0[strColnames[ii]].values, dtype=float)
        x = x.reshape(lDays, 1)
        y = y.reshape(lDays, 1)
       
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        yHat = regr.predict(x)
        # plot it as in the example at http://scikit-learn.org/
        # plt.scatter(x, y,  color='white')
        # plt.plot(x, yHat, color='blue', linewidth=3)
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()
        P1[strColnames[ii]] = yHat
    return P1

if __name__ == "__main__":
    # %matplotlib qt
    plt.close('all')
    currency = "EUR"
    metric = "Open"
    delta = dt.timedelta(days=356)
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
    
    
    
    P0 = get_Historical_Data(strStockFiat)
    
    #convert daily stock prices into daily log(x) returns
    # P0 = np.log2(P0*100)
    P0 = P0.tail(17)
    lNumEpochs = 711
    #Array to hold weight for each stock
    m_results = np.zeros( (4+lNumStocks-1,lNumEpochs) )
    lDays = ( endDate.date() -  startDate.date() ).days + 1
    r_Tot = np.log ( P0.iloc[-1] .div( P0.iloc[0] ) )
    r_Return_annual = 100 * (np.exp(r_Tot / lDays * 365) - 1)
    # P_log = get_Log_return(P0, r_Tot)
    P_log = get_Line_return(P0, r_Tot)
    std_max = 0.9   
    for ii in range(lNumEpochs):
        P0_weights = get_Random_weights(lNumStocks)
    
        D_t = P0 / P_log - 1
        std_Dt = D_t.std()
    
        #calculate portfolio return and volatility(standard deviation)
        ii_return  = np.sum(r_Return_annual * P0_weights)
        ii_std_dev = np.sum(std_Dt * P0_weights)
    
        #store results in results array
        m_results[0,ii] = ii_return
        m_results[1,ii] = ii_std_dev
        #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        # v1 = r_Return_annual * (1 - ( (std_Dt * P0_weights)/std_max)**0.2) * P0_weights
        v1 = ( r_Return_annual - 10*(std_Dt) )* P0_weights
    
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