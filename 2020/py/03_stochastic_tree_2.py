# Useful references
#
# The proper way to create a numpy array inside a for-loop
# http://akuederle.com/create-numpy-array-with-for-loop
#

import math
import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt

from gdl_finance import analytic

# Financial Parameters
S0     = 100.0   # initial stock level
K      = 100.0   # strike price
T      =   1.0   # time-to-maturity
r      =   0.05  # short rate
sigma  =   0.20  # volatility
delta  =   0.0   # dividend yield
payout =   1     # put option 

# Simulation Parameters
b  = 10                    # branching parameter (must be an even number!!!)
n  = 1                     # Monte Carlo reiterations
t  = [0.,1./3.,2./3.,1.]   # exercise opportunities
dt = t[1]-t[0]
df = math.exp(-r * dt)

european_option = analytic.BlackScholes(payout, S0, K, r, delta, sigma, T)

plx = []
ply = []
for n in range(100, 10002, 2):
    #
    # Random numbers generations. We use the function 'randn' which returns 
    # a sample (or samples) from the "standard normal" distribution.
    # If positive, int_like or int-convertible arguments are provided,
    # 'randn' generates an array of shape (d0, d1, ..., dn), filled
    # with random floats sampled from a univariate "normal" (Gaussian)
    # distribution of mean 0 and variance 1
    #
    z1  = np.random.randn(int(b/2),n)
    #
    # Stock Price Paths. The function 'cumsum' returns the cumulative sum 
    # of the elements along a given axis, in this case we want to sum along
    # the time axis (axis=0). Note that the first index of a numpy array 
    # by default refers to the row and the second index refers to the column
    
    #
    # ----- 1st exercise data -----------------------------------------------
    #
    S11 = S0 * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * z1)
    # antithetic
    S12 = S0 * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * (-z1))
    S1  = np.concatenate((S11, S12))
    #
    # ----- 2nd exercise data -----------------------------------------------
    #
    #
    # note that the first parameter is set to 0 since this is the
    # dimension along which we want to aggregate results
    # 
    S2  = np.empty((0, n))
    
    for i in range(0, b):
        z2  = np.random.randn(int(b/2),n)
        S21 = S1[i,:] * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * z2)
        # antithetic
        S22 = S1[i,:] * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * (-z2)) 
        S2_iter = np.concatenate((S21,S22))
    
        S2      = np.append(S2, S2_iter, axis=0)
        
    #    
    # HIGH ESTIMATOR -----------------------------------------------------------------------------
    #
    est_high = 0
    #
    # Calculate the value of an European Option starting at T-1
    # and maturity at T
    #
    final_payoff = analytic.BlackScholes(payout, S2, K, r, delta, sigma, 1./3.)
    est_1  = np.maximum(payout*(S2-K), final_payoff)
    
    cont_2 = [sum(est_1[i:i+b,:]) for i in range(0, np.size(est_1, 0), b)]
    cont_2 = np.divide(cont_2, b) * math.exp(-r*dt)
    est_2  = np.maximum(payout*(S1-K), cont_2)
        
    cont_3 = np.divide(np.sum(est_2, axis=0),b)* math.exp(-r*dt)
    est_high = np.maximum(payout*(S0-K), cont_3)
    #    
    # LOW ESTIMATOR -----------------------------------------------------------------------------
    #
    est_low = 0
    #
    # T2 estimation
    #
    eta_2   = np.zeros((b,n))
    est_4   = np.zeros((b,n))
    x       = range(0, b*b)
    for i in range(0, b):
        for k in range(0, b/2):
            y1 = np.asarray([b*i + k, b*i + k + b/2])
            y2 = np.delete(x, y1)
            cont_4 = np.sum([est_1[j] for j in y1], axis=0)/2
            cont_5 = np.sum([est_1[j] for j in y2], axis=0)/(b-2)
            for j in range(0,n):
                threshold = cont_5[j] * math.exp(-r*dt)
                exercise  = np.maximum(payout*(S1[i,j]-K),0)
                if exercise > threshold:
                    eta_2[k,j] = exercise
                else:
                    eta_2[k,j] = cont_4[j] * math.exp(-r*dt)
        est_4[i,:] = np.divide(np.sum(eta_2, axis=0),b/2)
    #
    # T1 estimation
    #
    eta_3   = np.zeros((b/2,n))
    x       = range(0, b)
    for k in range(0, b/2):
        y1 = np.asarray([k, k + b/2])
        y2 = np.delete(x, y1)
        cont_7 = np.sum([est_4[j] for j in y1], axis=0)/2
        cont_6 = np.sum([est_4[j] for j in y2], axis=0)/(b-2)
        for j in range(0,n):
            threshold = cont_6[j] * math.exp(-r*dt)
            exercise  = np.maximum(payout*(S0-K),0)
            if exercise > threshold:
                eta_3[k,j] = exercise
            else:
                eta_3[k,j] = cont_7[j] * math.exp(-r*dt)
    
    est_low = np.divide(np.sum(eta_3, axis=0),b/2)
    
    print('monte carlo sim  = ' + str(n))
    print('branching        = ' + str(b))

    point_estimate = 0.5*est_low +0.5*est_high
    
    std_err_h      = 1.96 * np.std(est_high)/np.sqrt(n)
    std_err_l      = 1.96 * np.std(est_low)/np.sqrt(n)
    
    est_high       = np.average(est_high)
    est_low        = np.average(est_low)
    point_estimate = np.average(point_estimate)
    
    print('black scholes    = ' + str(european_option))
    
    print('high  estimator  = ' + str(est_high))
    print('low   estimator  = ' + str(est_low))
    print('point estimator  = ' + str(point_estimate))

    print('95% Confidence Interval')
    print(str(est_low - std_err_l) + ' - ' + str(est_high + std_err_h))
    
    plx.append(n)
    ply.append(point_estimate)

if len(plx) > 1:
    y_inf = 0.95
    y_sup = 2.0 - y_inf
    y_inf = y_inf * european_option
    y_sup = y_sup * european_option
    y_del = (y_sup-y_inf)/10.0
    
    df = pd.DataFrame({'x':plx, 'y':ply})
    df.plot('x', 'y', kind='line', legend=None, color='blue')
    plt.hlines(np.average(ply), min(plx), max(plx), colors='red',   linewidth=2)
    plt.hlines(european_option, min(plx), max(plx), colors='green', linewidth=2)
    plt.title('Option Price Vs Monte Carlo Iterations (antithetic)', fontsize=14)
    plt.xlabel('Number of Simulations', fontsize=14)
    plt.ylim((y_inf , y_sup))
    plt.yticks(np.arange(y_inf, y_sup, y_del))
    plt.ylabel('Option Price', fontsize=14)
    plt.grid(True)
    plt.savefig('random_tree_2.png')
    plt.show()

    filename = 'dump_stochastic_tree_av.csv'
    df.to_csv(filename, index=False)
    
    print('average y = ' + str(np.average(ply)))
    print('It worked!')
