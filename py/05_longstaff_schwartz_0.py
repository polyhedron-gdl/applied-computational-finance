# Useful references
#
# The proper way to create a numpy array inside a for-loop
# http://akuederle.com/create-numpy-array-with-for-loop
#
# Note:
#
# - senza variabili antitetiche
#
import math
import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt

import BS

# np.random.seed(150000)
# Option Parameters
K     = 100.0   # strike price
delta =   0.0   # dividend yield 
T     =   1.00  # time-to-maturity
r     =   0.00  # short rate

# Asset Parameters
S0    = 120.0   # initial stock level
sigma =   0.20  # volatility

# Simulation Parameters
b  = 0                     # branching parameter (must be an even number!!!)
t  = [0.,1./3.,2./3.,1.]   # exercise opportunities
dt = t[1]-t[0]
df = math.exp(-r * dt)

european_option = BS.BlackScholes('C', S0, K, r, delta, sigma, T)
point_estimate  = 0
plx = []
ply = []
for b in range(10,10002,2):
    #    
    # PATH GENERATOR -----------------------------------------------------------------------------
    #
    # ----- 1st exercise data -----------------------------------------------
    #
    z1  = np.random.randn(b)
    #
    S1 = S0 * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * z1)
    #
    # ----- 2nd exercise data -----------------------------------------------
    #
    z2  = np.random.randn(b)
    #
    S2 = S1 * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * z2)
    #
    # Time T2 = T - 1
    #
    Yt2 = BS.BlackScholes('C', S2, K, r, delta, sigma, 1./3.)  
    Lt2 = np.maximum(np.maximum(S2 - K, 0), Yt2)
    #
    # Time T1 = T - 2
    #
    in_the_money=(S1-K>=0)
    Y = Lt2 * math.exp(-r*dt)
    X = np.zeros((b, 4))
    for i in range(0, b):
        bs_price = BS.BlackScholes('C', S1[i], K, r, delta, sigma, 2./3.)
        X[i,0] = K
        X[i,1] = S1[i]
        X[i,2] = bs_price
        X[i,3] = S1[i] * bs_price
            
    beta_psi    = np.linalg.inv(np.dot(X.T, X))
    beta_psi_v  = np.dot(X.T, Y)
    beta        = np.dot(beta_psi, beta_psi_v)
    
    cont        = np.dot(X, beta)
    
    Lt1 = Lt2 * math.exp(-r*dt)
    Yt1 = Yt2 * math.exp(-r*dt)
    for i in range(0,b):
        if max(S1[i]-K,0) > cont[i]:
            Lt1[i] = max(S1[i]-K,0)
            Yt1[i] = BS.BlackScholes('C', S0, K, r, delta, sigma, 2./3.)
    
    Lt0 = Lt1 * math.exp(-r*dt)        
    Yt0 = Yt1 * math.exp(-r*dt)    
    
    point_estimate = np.average(Lt0)    
        
    print('branching        = ' + str(b))
    print('point estimator  = ' + str(point_estimate))
    print('black scholes    = ' + str(european_option))
        
    plx.append(b)
    ply.append(point_estimate)

print('average y = ' + str(np.average(ply)))

if len(plx) > 1:
    y_inf = 0.8
    y_sup = 2.0 - y_inf

    df = pd.DataFrame({'x':plx, 'y':ply})
    #
    ax = df.plot('x', 'y', kind='scatter', s=2)
    ax.set_xlim((min(plx), max(plx)))
    ax.set_ylim((y_inf * european_option, y_sup * european_option))
    plt.hlines(np.average(ply)  , min(plx), max(plx), colors='red'  , linewidth=2)
    plt.hlines(european_option  , min(plx), max(plx), colors='green', linewidth=2)
    plt.title('Option Price Vs Monte Carlo Iterations', fontsize=14)
    plt.xlabel('Number of Simulations', fontsize=14)
    plt.ylabel('Option Price', fontsize=14)
    plt.grid(True)
    plt.savefig('plot_longstaff_schwartz_0.png')
    plt.show()
    #
    df.to_csv('dump_longstaff_schwartz_0.csv', index=False)

print('It worked!')