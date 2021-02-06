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
sigma =   0.4   # volatility
navg  =   24
#
# Simulation Parameters -------------------------------------------------------------
#
b     = 0       # number of paths
M     = 25      # number of points for each path
dt    = float(T) /float( M)
df    = math.exp(-r * T)
#
# -----------------------------------------------------------------------------------
#
x   = []
y1  = []
y2  = []
y3  = []
for b in range(2, 10002, 2):
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
    #--------------------------------------------------------------------------------
    #
    # Simple Montecarlo without variance reduction
    #
    S1 = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
        + sigma * math.sqrt(dt) * z1, axis=0))
    S1 = np.insert(S1, 0, S0, axis=0)

    avg  = np.average(S1[-navg:-1,:], axis=0)
    gavg = gmean     (S1[-navg:-1,:], axis=0)
    
    opt_asian_1      = np.maximum(avg   - K, 0) * df
    std_dev_1        = np.std(opt_asian_1)/math.sqrt(float(b))
    opt_asian_1      = np.average(opt_asian_1)
    #
    #--------------------------------------------------------------------------------
    #
    # Antithetic Variable 
    #
    z2  = np.random.randn(M, int(b/2))  
    #
    S21 = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * z2, axis=0))
    S22 = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * (-z2), axis=0))
    S21 = np.insert(S21, 0, S0, axis=0)
    S22 = np.insert(S22, 0, S0, axis=0)

    avg1  = np.average(S21[-navg:-1,:], axis=0)
    avg2  = np.average(S22[-navg:-1,:], axis=0)
    
    opt_asian_21 = np.maximum(avg1 - K, 0) * df
    opt_asian_22 = np.maximum(avg2 - K, 0) * df

    opt_asian_2 = 0.5 * (opt_asian_21 + opt_asian_22)
    std_dev_2   = np.std(opt_asian_2)/math.sqrt(float(b))
    opt_asian_2 = np.average(opt_asian_2)
    #
    #--------------------------------------------------------------------------------
    #
    # Control Variate 
    #
    opt_estimate    = np.maximum(avg   - K, 0) * df 
    control_variate = np.maximum(gavg  - K, 0) * df
    
    covariance      = np.cov(control_variate, opt_estimate)
    variance        = np.var(control_variate)
    
    beta            = covariance[0][1]/variance
    
    opt_asian_g_teo = BS.AsianGeometric('C', S0, K, r, delta, sigma, T, navg)
    opt_asian_3     = opt_estimate - beta*(control_variate - opt_asian_g_teo)
    std_dev_3       = np.std(opt_asian_3)/math.sqrt(float(b))
    opt_asian_3     = np.average(opt_asian_3)
    
    x.append(b)
    y1.append(opt_asian_1)
    y2.append(opt_asian_2)
    y3.append(opt_asian_3)

    print('montecarlo simulations  : ' + str(b))
    print('asian montecarlo        : ' + str(opt_asian_1)) 
    print('std asian               : ' + str(std_dev_1))
    print('asian montecarlo av     : ' + str(opt_asian_2)) 
    print('std asian av            : ' + str(std_dev_2))
    print('asian montecarlo cv     : ' + str(opt_asian_3)) 
    print('std asian cv            : ' + str(std_dev_3)) 
    print('beta                    : ' + str(beta))
#
# -----------------------------------------------------------------------------------
#
if len(x) > 1:
    
    y_inf = 0.75
    y_sup = 2.0 - y_inf
    
    df1 = pd.DataFrame({'x':x, 'y':y1})
    ax = df1.plot('x', 'y', kind='scatter', s=2)
    ax.set_xlim((min(x), max(x)))
    ax.set_ylim((y_inf * opt_asian_1, y_sup * opt_asian_1))
    
    plt.hlines(np.average(y1), min(x), max(x), colors='red',        linewidth=1.5)
    plt.title('Option Price Vs Monte Carlo Iterations (Simple MC)', fontsize=12)
    plt.xlabel('Number of Simulations',                             fontsize=12)
    plt.ylabel('Option Price',                                      fontsize=12)
    plt.grid(True)
    
    df2 = pd.DataFrame({'x':x, 'y':y2})
    ax = df2.plot('x', 'y', kind='scatter', s=2)
    ax.set_xlim((min(x), max(x)))
    ax.set_ylim((y_inf * opt_asian_1, y_sup * opt_asian_1))
    
    plt.hlines(np.average(y2), min(x), max(x), colors='red',        linewidth=1.5)
    plt.title('Option Price Vs Monte Carlo Iterations (AV MC)',     fontsize=12)
    plt.xlabel('Number of Simulations',                             fontsize=12)
    plt.ylabel('Option Price',                                      fontsize=12)
    plt.grid(True)
    
    df3 = pd.DataFrame({'x':x, 'y':y3})
    ax = df3.plot('x', 'y', kind='scatter', s=2)
    ax.set_xlim((min(x), max(x)))
    ax.set_ylim((y_inf * opt_asian_1, y_sup * opt_asian_1))
    
    plt.hlines(np.average(y3), min(x), max(x), colors='red',        linewidth=1.5)
    plt.title('Option Price Vs Monte Carlo Iterations (CV MC)',     fontsize=12)
    plt.xlabel('Number of Simulations',                             fontsize=12)
    plt.ylabel('Option Price',                                      fontsize=12)
    plt.grid(True)

    plt.show()

    filename = 'dump_variance_reduction_1.csv'
    df1.to_csv(filename, index=False)

