import math
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
import scipy

from scipy.stats.mstats import gmean

import BS

np.random.seed(1234)
#
# Model Parameters -----------------------------------------------------------------
#
S0    = 100.0   # initial stock level
K     =  70.0   # strike price
T     =   1.0   # time-to-maturity
r     =   0.01  # short rate
delta =   0.0   # dividend yield
sigma =   0.2   # volatility
navg  =  12
#
# Simulation Parameters -------------------------------------------------------------
#
b = 0       # number of paths
M = 25      # number of points for each path
dt = float(T) /float( M)
df = math.exp(-r * T)
#
# -----------------------------------------------------------------------------------
#
x   = []
y1  = []
y2  = []
for b in range(1, 2500, 1):
    #
    # Random numbers generations. We use the function 'randn' which returns 
    # a sample (or samples) from the "standard normal" distribution.
    # If positive, int_like or int-convertible arguments are provided,
    # 'randn' generates an array of shape (d0, d1, ..., dn), filled
    # with random floats sampled from a univariate "normal" (Gaussian)
    # distribution of mean 0 and variance 1
    #
    z1  = np.random.randn(M, b) 
    #
    # Stock Price Paths. The function 'cumsum' returns the cumulative sum 
    # of the elements along a given axis, in this case we want to sum along
    # the time axis (axis=0)
    #
    S1 = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
        + sigma * math.sqrt(dt) * z1, axis=0))
    S1 = np.insert(S1, 0, S0, axis=0)

    avg  = np.average(S1[-navg:-1,:], axis=0)
    gavg = gmean     (S1[-navg:-1,:], axis=0)
    last = S1[-1,:]
    
    opt_call_est    = np.maximum(last  - K, 0) * df
    std_call        = np.std(opt_call_est)/math.sqrt(float(b))
    opt_call_est    = np.average(np.maximum(last  - K, 0)) * df
    
    opt_asian       = np.maximum(avg   - K, 0) * df
    std_asian       = np.std(opt_asian)/math.sqrt(float(b))
    opt_asian       = np.average(np.maximum(avg   - K, 0)) * df
    
    opt_call_teo    = BS.BlackScholes('C', S0, K, r, delta, sigma, T)
    opt_asian_g     = np.average(np.maximum(gavg  - K, 0)) * df
    opt_asian_g_teo = BS.AsianGeometric('C', S0, K, r, delta, sigma, T, 25)

    print('montecarlo simulations  : ' + str(b))
    print('call black & scholes    : ' + str(opt_call_teo))
    print('call montecarlo         : ' + str(opt_call_est)) 
    print('std call                : ' + str(std_call))
    print('asian montecarlo        : ' + str(opt_asian)) 
    print('std asian               : ' + str(std_asian))
    print('asian gavg montecarlo   : ' + str(opt_asian_g)) 
    print('asian gavg analytic     : ' + str(opt_asian_g_teo)) 


#t = np.linspace(0, T, M+1)

# plotting the first n paths

#paths_1 = S1[:,0:50]
#fig = plt.figure(figsize=(15,5))
#axis = fig.add_subplot(121, autoscale_on=False, xlim=(0,T), ylim=(50,150))
#plt.plot(t, paths_1)    

#plt.show()    
    
