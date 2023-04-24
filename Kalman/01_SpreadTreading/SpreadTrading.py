# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 22:02:17 2022

@author: perletti
"""

import numpy as np


# class KalmanFilter(object):
# 	def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

# 		if(F is None or H is None):
# 			raise ValueError("Set proper system dynamics.")

# 		self.n = F.shape[1]
# 		self.m = H.shape[1]

# 		self.F = F
# 		self.H = H
# 		self.B = 0 if B is None else B
# 		self.Q = np.eye(self.n) if Q is None else Q
# 		self.R = np.eye(self.n) if R is None else R
# 		self.P = np.eye(self.n) if P is None else P
# 		self.x = np.zeros((self.n, 1)) if x0 is None else x0

# 	def predict(self, u = 0):
# 		self.x = np.dot(self.F, self.x) + np.dot(self.B, u) 
# 		self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
# 		return self.x

# 	def update(self, z):
# 		y = z - np.dot(self.H, self.x)
# 		S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
# 		K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
# 		self.x = self.x + np.dot(K, y)
# 		I = np.eye(self.n)
# 		self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
# 			(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
def kf_predict(X, P, A, Q, B, U):
		X = np.dot(A, X) + np.dot(B, U)
		P = np.dot(A, np.dot(P, A.T)) + Q
		return(X,P)
    
def kf_update(X, P, Y, H, R):
        IM = np.dot(H, X)
        IS = R + np.dot(H, np.dot(P, H.T))
        K = np.dot(P, np.dot(H.T, np.linalg.inv(IS)))
        X = X + np.dot(K, (Y-IM))
        P = P - np.dot(K, np.dot(IS, K.T))
        LH = gauss_pdf(Y, IM, IS)
        return (X,P,K,IM,IS,LH) 
    
def gauss_pdf(X, M, S):
     if M.shape[1] == 1:
         DX = X - np.tile(M, X.shape[1])
         E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0 )
         E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
         P = np.exp(-E)
     elif X.shape[1] == 1:
         DX = np.tile(X, M.shape[1])- M
         E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
         E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
         P = np.exp(-E)
     else:
         DX = X-M
         E = 0.5 * np.dot(DX.T, np.dot(np.linalg.inv(S), DX))
         E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
         P = np.exp(-E)
     return (P[0],E[0])
     
def example():
    import pandas as pd
    import pandas_datareader as web
    #import yfinance as yf
    import matplotlib.pyplot as plt
    #import mplfinance as mpf 

    import datetime as dt
    currency = "USD"
    
    start = dt.date(2022,5, 5)
    end =  dt.datetime.now()
    today = end.date()
    
    
    strBase ='ADA'
    strCsvFile = f'{strBase}-{currency}-{start}-{today}.csv'
    try:
        data_BaseLine = pd.read_csv(strCsvFile, parse_dates=[0], dayfirst=True, index_col = 'Date')
    except OSError:
        data_BaseLine = web.DataReader(f'{strBase}-{currency}', "yahoo", start, end)
        data_BaseLine.to_csv(strCsvFile)
    
    crypto = [
               # 'CRO', 'BNB', 'BTC','XRP',
                'CRO' ,'ETH' , 
               # 'VVS', 'BIFI','ADA',
               # 'SOL', 'LUNA1', 'AVAX' , 'DOT' , 
               # 'CRO' , 'WBTC',  'AVAX' , 
               # 'JUV', 'BONDLY'
              ]
    
    for ticker in crypto:
        strCsvFile = f'{ticker}-{currency}-{start}-{today}.csv'
        
        try:
            data_Raw = pd.read_csv(strCsvFile, parse_dates=[0], dayfirst=True, index_col = 'Date')
        except OSError:
            data_Raw = web.DataReader(f'{ticker}-{currency}', "yahoo", start, end)
            data_Raw.to_csv(strCsvFile)
            
            
        data = pd.DataFrame() # combined data frame
                # Divide the DataFrame1 elements by the elements of DataFrame2
        if 0:
            data['Open']      = data_Raw['Open'].div(data_BaseLine['Open']);
            data['High']      = data_Raw['High'].div(data_BaseLine['High']);
            data['Low']       = data_Raw['Low'].div(data_BaseLine['Low']);
            data['Adj Close'] = data_Raw['Adj Close'].div(data_BaseLine['Adj Close']);
            data['Volume']    = data_Raw['Volume'].div(data_BaseLine['Volume']);
        else :
            data['Open']      = data_Raw['Open']
            data['High']      = data_Raw['High']
            data['Low']       = data_Raw['Low']
            data['Adj Close'] = data_Raw['Adj Close']
            data['Volume'] = data_Raw['Volume']
            
        data.dropna(inplace = True) # not a number values
        #data2 =  yf.download('BTC',start,interval="1h")
        for ii in range(1):
            if ii==4:
                rule_lookup={
                    0:'W-MON',
                    1:'W-TUE',
                    2:'W-WED',
                    3:'W-THU',
                    4:'W-FRI',
                    5:'W-SAT',
                    6:'W-SUN'
                }
                
                # get the proper rule which ends on the last date in the index
                rule = rule_lookup[data.index[-1].weekday()] 
                print(f"=> resampling using rule: {rule}")

                data = data.resample(rule=rule).agg({'Open': 'first', 
                                              'High': 'max', 
                                              'Low': 'min', 
                                              'Adj Close': 'last',
                                              'Volume': 'sum'})
                data_BaseLine = data_BaseLine.resample(rule=rule).agg({'Open': 'first', 
                                              'High': 'max', 
                                              'Low': 'min', 
                                              'Adj Close': 'last',
                                              'Volume': 'sum'})
            
            data_BaseLine['Log'] = data_BaseLine['Adj Close']
            data['Log']          = data['Adj Close']
            data_BaseLine['Log'] = data_BaseLine['Log'].apply(lambda x: np.log(1+x))
            data['Log']          = data['Log'].apply(lambda x: np.log(1+x))
            
            # time step of mobile movement
            # Initialization of state matrices
            
            A = np.array([ [1 , 0] , [0, 1] ] )
            X = np.array([ [1  ] , [1   ] ] )
            B = np.eye(X.shape[0])
            U = np.zeros((X.shape[0],1)) 
            Q = np.array([ [.1, 0] , [0, 1 ]] )
            
            gamma = 1.01 # perfect measure
            mu = 1
            y2_m = 107
            y1_m = mu  +  gamma * y2_m # + np.random.randn(1) # dirten measure

            # Measurement matrices
            H = np.array([[1 , y2_m]])
            Y = np.array([[y1_m]])
        
            yStd = 3
            R = np.eye(Y.shape[0])*yStd**2
            # Signal Processing, Robust Estimation, Kalman, Optimization - 42:25 
            
            P = np.diag((10, 10))
            
            N_iter = len(data) # Number of iterations in Kalman Filter
            vY1_m = [] # vectr y1 measured
            vY2_m = [] # vectr y2 measured
            vY1_p = [] # vectr Predictions
            vX0_o = [] # vectr Predictions
            vX1_o = [] # vectr Predictions
            vZ    = []
            vSumY1 = [] # coin Y1
            vSumY2  = []# coin Y2
            vCash  = []# total value portfolio
            lCumSumY1 = 500/data_BaseLine['Adj Close'][0]
            lCumSumY2 = 500/data['Adj Close'][0]
            lCash = 20 #cash to invest in trading
            THR = 0.0001
            # Applying the Kalman Filter
            for i in np.arange(2, N_iter):

                # y2_m = float( np.random.randn(1)*1 +12 + 5*np.sin(6.28*i/51)) # dirten measure
                # y1_m = float(mu + i/40 +  gamma * y2_m + 1*np.sin(6.28*i/5555)) # dirten measure
                y1_m = data_BaseLine['Log'][i]
                y2_m = data['Log'][i]
                H = np.array([[1 , y2_m]])
                y1_p = float( H.dot(X) ) 
                (X, P) = kf_predict(X, P, A, Q, B, U)

                (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
                if i < N_iter:
                    Y = np.array([ [y1_m ] ]) 
                else:
                    Y = np.array([ [y1_p ] ])
                zScore =   y1_m - X[1] * y2_m - X[0]
                vY1_m.append( ( y1_m ))
                vY2_m.append( ( y2_m ))
                vY1_p.append( float(zScore))
                vX0_o.append( float( X[0] ))
                vX1_o.append( float( X[1] ))
                
                kY1 = 0
                if (zScore > THR):
                    kY1 = +1
                elif(zScore < -THR):
                    kY1 = -1
                else:
                    kY1 = 0
                
                if  i > 50:
                    lCumSumY1 =  lCumSumY1 - 1.0 * kY1 * lCash / data_BaseLine['Adj Close'][i]
                    lCumSumY2 =  lCumSumY2 + 1.0 * kY1 * lCash / data['Adj Close'][i]
                
                vSumY1.append( lCumSumY1 * data_BaseLine['Adj Close'][i] )
                vSumY2.append( lCumSumY2 * data['Adj Close'][i] )
                vCash.append(vSumY1[i-2] + vSumY2[i-2])
            lLast = (start - today).days+10#-100
            ax1 = plt.subplot(311)
            plt.plot( vY1_m[lLast:]/np.max(vY1_m), label = 'y1 meas')
            plt.plot( vY2_m[lLast:]/np.max(vY2_m), label = 'y2 meas')

            fStd = np.std(vY1_p[100:]) # 33% data
            ax1.axhline(+0.5,linestyle='--', alpha =0.5, color = '#00aaFF')
            ax1.axhline(-0.5,linestyle='--', alpha =0.5, color = '#FFaa00')
            plt.fill_between( range(len(vY1_p[lLast:])), np.tanh(vY1_p[lLast:]/fStd), label = 'residual', step="pre", alpha=0.4)
            plt.legend(loc='center left')
            plt.title(  ticker + '/' + strBase )
            ax1 = plt.subplot(312, sharex=ax1)
            plt.plot( vX0_o[lLast:], label = 'mu Prediction')
            plt.plot( vX1_o[lLast:], label = 'gamma Prediction')
            # vY1_p[:10] = 0
            fStd = np.std(vY1_p[100:])*0.5 # 33% data
            ax1.axhline(+fStd,linestyle='--', alpha =0.5, color = '#00aaFF')
            plt.plot( vY1_p[lLast:], label = 'residual')
            ax1.axhline(-fStd,linestyle='--', alpha =0.5, color = '#FFaa00')
            plt.legend(loc='center left')
    
            ax3 = plt.subplot(313, sharex=ax1)
            plt.plot( vSumY1[lLast:], label = f'{strBase}')
            plt.plot( vSumY2[lLast:], label = f'{ticker}')
            plt.plot( vCash, label = 'Cash',linestyle='--', alpha =0.5, color = '#FFaa00')

            # plt.plot( vSumG, label = 'ySUM')

            plt.legend(loc='center left')
            plt.show()
            # dt = 1.0/60
            # STEPS = 30
            # varQ = 1
            # F = np.array([[1, dt, 0],
            #               [0, 1, dt],
            #               [0, 0, 1]])
            
            # H = np.array([1, 0, 0]).reshape(1, 3)
        
            # Q = np.array([
            #     [varQ, varQ, 0.0], 
            #     [varQ, varQ, 0.0],
            #     [0.0, 0.0, 0.0]])
            
            # R = np.array([10]).reshape(1, 1)
        
            # x = np.linspace(-10, 10, STEPS)
            # y_real = - (0.1*x - 2)  + np.random.normal(0, 2, STEPS)
        
            # kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
            # predictions = []
        
            # for ii in range(STEPS):
            #     z = y_real[ii]
            #     predictions.append(np.dot(H,  kf.predict())[0])
            #     if ii % 5 == 0:
            #         kf.update(z)
        
            # import matplotlib.pyplot as plt
            # plt.plot(range(len(y_real)), y_real, label = 'Measurements')
            # plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
            # plt.legend()
            # plt.show()


if __name__ == '__main__':
	example()