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
    #time step of mobile movement
    # Initialization of state matrices

    A = np.array([ [1 , 0] , [0, 1] ] )
    X = np.array([ [1  ] , [1   ] ] )
    B = np.eye(X.shape[0])
    U = np.zeros((X.shape[0],1)) 
    Q = np.array([ [1 , 0] , [0, 1] ] )
    
    
    gamma = 1.01 # perfect measure
    mu = 1
    y2_m = 10
    y1_m = mu  +  gamma * y2_m # + np.random.randn(1) # dirten measure

    # Measurement matrices
    H = np.array([[1 , y2_m]])
    Y = np.array([[y1_m]])

    
    yStd = 1
    R = np.eye(Y.shape[0])*yStd**2
    # Signal Processing, Robust Estimation, Kalman, Optimization - 42:25 
    
    P = np.diag((10, 10))
    
   
    
    N_iter = 300 # Number of iterations in Kalman Filter
    vMeas = [] # vectr Measurements
    vY1_m = [] # vectr y1 measured
    vY2_m = [] # vectr y2 measured
    vY1_p = [] # vectr Predictions
    vX0_o = [] # vectr Predictions
    vX1_o = [] # vectr Predictions
    vZ    = []
    # Applying the Kalman Filter
    for i in np.arange(0, N_iter):
        (X, P) = kf_predict(X, P, A, Q, B, U)
        
        (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
        
        y2_m = float( np.random.randn(1)*1 +12 + 5*np.sin(6.28*i/51)) # dirten measure
        y1_m = float(mu + i/40 +  gamma * y2_m + 1*np.sin(6.28*i/5555)) # dirten measure
    
        H = np.array([[1 , y2_m]])
        y1_p = float( H.dot(X) ) 
        
        if i < N_iter*0.99:
            Y = np.array([ [y1_m ] ]) 
        else:
            Y = np.array([ [y1_p ] ]) 
        
        vY1_m.append( ( y1_m ))
        vY2_m.append( ( y2_m ))
        vY1_p.append( float( y1_m - X[1] * y2_m - X[0]))
        vX0_o.append( float( X[0] ))
        vX1_o.append( float( X[1] ))
        vZ.append( y1_m - float( X[1] )*y2_m )
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211)
    plt.plot( vY1_m, label = 'y1 meas')
    plt.plot( vY2_m, label = 'y2 meas')

    ax1 = plt.subplot(212)
    plt.plot( vX0_o, label = 'mu Prediction')
    plt.plot( vX1_o, label = 'gamma Prediction')
    plt.plot( vY1_p, label = 'residual',linestyle=':')
    # plt.plot( vZ, label = 'Spread')
    plt.legend()
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