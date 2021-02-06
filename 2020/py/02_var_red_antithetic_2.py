import math
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt

from gdl_finance.analytic import BlackScholes

# np.random.seed(1234)
#
# Model Parameters -----------------------------------------------------------------
#
S0     =  90.0   # initial stock level
K      = 100.0   # strike price
T      =   1.0   # time-to-maturity
r      =   0.01  # short rate
delta  =   0.0   # dividend yield
sigma  =   0.2   # volatility
payout = -1
#
# Simulation Parameters -------------------------------------------------------------
#
b = 0      # number of paths
M = 3      # number of points for each path
dt = float(T) /float( M)
df = math.exp(-r * T)
#
# -----------------------------------------------------------------------------------
#
x   = []
y1  = []
y2  = []
for b in range(2, 1002, 2):
    option_price = BlackScholes(payout, S0, K, r, delta, sigma, T)
    #
    # Random numbers generations. We use the function 'randn' which returns 
    # a sample (or samples) from the "standard normal" distribution.
    # If positive, int_like or int-convertible arguments are provided,
    # 'randn' generates an array of shape (d0, d1, ..., dn), filled
    # with random floats sampled from a univariate "normal" (Gaussian)
    # distribution of mean 0 and variance 1
    #
    z1  = np.random.randn(M, b) 
    z1 = np.divide(z1 - np.mean(z1), np.std(z1))
    #
    # Stock Price Paths. The function 'cumsum' returns the cumulative sum 
    # of the elements along a given axis, in this case we want to sum along
    # the time axis (axis=0)
    #
    S1 = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
        + sigma * math.sqrt(dt) * z1, axis=0))
    S1 = np.insert(S1, 0, S0, axis=0)
    
    option_price_est_1 = np.maximum(payout*(S1[-1] - K), 0) * df
    std_dev_1          = np.std(option_price_est_1)
    option_price_est_1 = np.average(option_price_est_1)
    #
    #--------------------------------------------------------------------------------
    #
    z2  = np.random.randn(M, int(b/2))  
    z2 = np.divide(z2 - np.mean(z2), np.std(z2))
    #
    # Stock Price Paths. The function 'cumsum' returns the cumulative sum 
    # of the elements along a given axis, in this case we want to sum along
    # the time axis (axis=0)
    #
    S21 = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * z2, axis=0))
    S22 = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * (-z2), axis=0))
    S21 = np.insert(S21, 0, S0, axis=0)
    S22 = np.insert(S22, 0, S0, axis=0)
    
    option_price_est_21 = np.maximum(payout*(S21[-1] - K), 0) * df
    option_price_est_22 = np.maximum(payout*(S22[-1] - K), 0) * df

    option_price_est_2 = 0.5 * (option_price_est_21 + option_price_est_22)
    std_dev_2          = np.std(option_price_est_2)
    option_price_est_2 = np.average(option_price_est_2)
    #
    #--------------------------------------------------------------------------------
    #
    x.append(b)
    y1.append(option_price_est_1)
    y2.append(option_price_est_2)
    
    print('Simulation number    = ' + str(b))
    print('BS price             = ' + str(option_price))
    print('Simple     MC        = ' + str(option_price_est_1) + ' +/- ' + str(1.96 * std_dev_1/math.sqrt(b)))
    print('Antithetic MC        = ' + str(option_price_est_2) + ' +/- ' + str(1.96 * std_dev_2/math.sqrt(b)))
    print('Simple     MC std    = ' + str(std_dev_1))
    print('Antithetic MC std    = ' + str(std_dev_2))

if len(x) > 1:
    
    y_inf = 0.75
    y_sup = 2.0 - y_inf
    
    df1 = pd.DataFrame({'x':x, 'y':y1})
    ax = df1.plot('x', 'y', kind='scatter', s=2)
    ax.set_xlim((min(x), max(x)))
    ax.set_ylim((y_inf * option_price, y_sup * option_price))
    
    plt.hlines(np.average(y1), min(x), max(x), colors='red',        linewidth=1.5)
    plt.hlines(option_price,   min(x), max(x), colors='yellow',     linewidth=1.5)
    plt.title('Option Price Vs Monte Carlo Iterations (Simple MC)', fontsize=12)
    plt.xlabel('Number of Simulations',                             fontsize=12)
    plt.ylabel('Option Price',                                      fontsize=12)
    plt.grid(True)
    plt.savefig('var_red_antithetic_2_simple.png')
    
    df2 = pd.DataFrame({'x':x, 'y':y2})
    ax = df2.plot('x', 'y', kind='scatter', s=2)
    ax.set_xlim((min(x), max(x)))
    ax.set_ylim((y_inf * option_price, y_sup * option_price))
    
    plt.hlines(np.average(y2), min(x), max(x), colors='red',        linewidth=1.5)
    plt.hlines(option_price,   min(x), max(x), colors='yellow',     linewidth=1.5)
    plt.title('Option Price Vs Monte Carlo Iterations (AV MC)',     fontsize=12)
    plt.xlabel('Number of Simulations',                             fontsize=12)
    plt.ylabel('Option Price',                                      fontsize=12)
    plt.grid(True)
    plt.savefig('var_red_antithetic_2_anti.png')

    plt.show()

    filename = 'dump_variance_reduction_1.csv'
    df1.to_csv(filename, index=False)

print('It worked!')
