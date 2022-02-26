# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:38:24 2022
https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
@author: perletti
"""
import pandas as pd
from scipy.stats import kurtosis, skew
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
    weight_P0 = np.array(np.random.randint(0,13,lNumStocks)).astype(float)
    #rebalance weights to sum to 1
    weight_P0 = weight_P0
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

def get_Poly_return(P0, lPolyVal):

    P1 = P0.copy()
    strColnames = P1.columns
    fig2, ax1 = plt.subplots()
    for ii in range(len(strColnames)):
        x=np.array(P0.index.values, dtype=float)
        y=np.array(P0[strColnames[ii]].values, dtype=float)
        model = np.poly1d(np.polyfit(x, y, int(lPolyVal)))
        
        #add fitted polynomial line to scatterplot
        polyline = np.linspace(x[0], x[-1], 50)
        ax1.scatter(x, y)
        ax1.plot(polyline, model(polyline))

        yHat = model(x)
        # plot it as in the example at http://scikit-learn.org/
        # plt.scatter(x, y,  color='white')
        # plt.plot(x, yHat, color='blue', linewidth=3)
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()
        P1[strColnames[ii]] = yHat
    plt.show()
    return P1

def get_cara_score(P0):
    P0_ret = P0.pct_change() #.apply(lambda x: np.log(1+x))
    cara = pd.DataFrame()
    cara['mean'] = P0_ret.mean()
    cara['std']  = P0_ret.std()
    cara['skew'] = P0_ret.skew()
    cara['kurt'] = P0_ret.kurtosis()
    # constant absolute risk aversion
    theta = float(5)
    cara['score'] = cara['mean'] 
    + 0.5 * theta * cara['std']**2 
    + theta**2 * cara['std']**3 * cara['skew'] / 6
    - theta**3 * cara['std']**4 * ( cara['kurt'] - 3 ) / 720
    return cara

def score_exec(P0, lNumEpochs):
    lNumStocks = len(P0.columns)
    lDays = len(P0)
    #Array to hold weight for each stock
    m_results = np.zeros( (5+lNumStocks-1,lNumEpochs) )
    r_Tot = np.log ( P0.iloc[-1].div( P0.iloc[0] ) )
    r_Return_annual = 100 * (np.exp(r_Tot / lDays * 365) - 1)
    Cara = get_cara_score(P0)
    
    # P_log = get_Log_return(P0, r_Tot)
    P_log = get_Poly_return(P0, 1)
    
    D_t = P0 / P_log - 1
    std_Dt = D_t.std()
    std_max = std_Dt.max()*1.6
    for ii in range(lNumEpochs):
        P0_weights = get_Random_weights(lNumStocks)

        #calculate portfolio return and volatility(standard deviation)
        ii_return  = np.sum(r_Return_annual * P0_weights)
        # ii_cara =  np.sum(r_Return_annual / std_Dt * P0_weights)
        ii_cara =  np.sum( Cara['score'] * P0_weights )
        # ii_vwr = np.sum(( r_Return_annual - (std_Dt)*100 )* P0_weights)
        ii_vwr1 = np.sum(
            r_Return_annual * P0_weights * 
            ( 1 - ( (std_Dt ) / std_max)**7 ) )
        ii_vwr2 = np.sum(
            r_Return_annual * P0_weights * 
            ( 1 - ( (std_Dt ) / std_max)**0.1 ) )
        #store results in results array
        m_results[0,ii] = ii_return
        m_results[1,ii] = ii_cara
        m_results[2,ii] = ii_vwr1
        m_results[3,ii] = ii_vwr2
        #iterate through the weight vector and add data to results array
        for jj in range(lNumStocks):
            m_results[jj+4,ii] = P0_weights[jj]
            
    # print(kurtosis(x, fisher=False))
    # print(skew(x, bias=False))
    
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
        strTitle += (':' + str(r_Tot[strTitle]))
        ax1.set_title(strTitle, color='white')   
    return m_results



if __name__ == "__main__":
    # %matplotlib qt
    plt.close('all')
    currency = "EUR"
    metric = "Open"
    delta = dt.timedelta(days=368)
    startDate = dt.datetime.now() - delta
    endDate =  dt.datetime.now()
    
    strStock = [ 
    'BTC', 'ETH', 'BNB', 'XRP' , 'USDC' , 'SOL' ,'CRO', 'AVAX'
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
    lDays = 11
    P0 = P0.tail(int(lDays))
    
    lNumEpochs = 1711
    
    
    m_results = score_exec(P0, lNumEpochs)
   
    strColStocks = P0.columns
    vColumns = ['ret','cara','vwr','joker']
    vColumns.extend(strColStocks)
    results_frame = pd.DataFrame(m_results.T, columns= vColumns )
    #locate position of portfolio with highest VWR Ratio
    r_Max_VWR = results_frame.iloc[results_frame['ret'].idxmax()]
    print("\n-----------------RET\n", r_Max_VWR)
    #locate positon of portfolio with minimum standard deviation
    r_Min_VWR = results_frame.iloc[results_frame['cara'].idxmax()]
    print("\n-----------------CARA\n", r_Min_VWR)
    #locate positon of portfolio with minimum standard deviation
    r_Min_std = results_frame.iloc[results_frame['vwr'].idxmax()]
    print("\n-----------------VWR\n", r_Min_std)
    #locate positon of portfolio with minimum standard deviation
    r_Jolly = results_frame.iloc[results_frame['joker'].idxmax()]
    print("\n-----------------JKR\n", r_Jolly)
    #create scatter plot coloured by Sharpe Ratio
    fig1, ax0 = plt.subplots()
    ax0 = plt.scatter(results_frame.cara,results_frame.vwr,c = results_frame.ret,cmap='RdYlBu')
    plt.scatter(r_Max_VWR[1], r_Max_VWR[0],color='y', marker='^', s=180 , alpha=0.5)
    plt.scatter(r_Min_VWR[1], r_Min_VWR[0],color='r', marker='v', s=150 , alpha=0.5)
    plt.scatter(r_Min_std[1], r_Min_std[0],color='g', marker='s', s=120  , alpha=0.5)
    plt.scatter(r_Jolly  [1], r_Jolly  [0],color='b', marker='x', s=90  , alpha=0.5)
    plt.xlabel('cara')
    plt.ylabel('vwr')
    plt.xticks(rotation=60)
    plt.grid(True, color = '#555555')
    plt.colorbar()
    plt.show()
    # weights = weights*100
    # print ("Weights to invest in ", stocks , "respectively are")
    # print (weights)
    

    