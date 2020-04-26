# Note:
#
# - il metodo della variabile di controllo non funziona
# - utilizzare il fancy indexing per la selezione dei valori
#   da utilizzare per exercise e continuation
#
import math
import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt
import BS

from scipy.stats    import norm 
from sklearn        import linear_model

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
b  = 40                     # branching parameter (must be an even number!!!)
n  = 1                     # Monte Carlo reiterations
t  = [0.,1./3.,2./3.,1.]   # exercise opportunities
dt = t[1]-t[0]
df = math.exp(-r * dt)

drift = (r - delta - .5*sigma**2) * dt

x = []
y = []

est_high        = 0
est_low         = 0
point_estimate  = 0

for n in range(100,101):
    european_option = BS.BlackScholes('C', S0, K, r, delta, sigma, T)
    #    
    # PATH GENERATOR -----------------------------------------------------------------------------
    #
    # Random numbers generations. We use the function 'randn' which returns 
    # a sample (or samples) from the "standard normal" distribution.
    # If positive, int_like or int-convertible arguments are provided,
    # 'randn' generates an array of shape (d0, d1, ..., dn), filled
    # with random floats sampled from a univariate "normal" (Gaussian)
    # distribution of mean 0 and variance 1
    #
    # ----- 1st exercise data -----------------------------------------------
    #
    z1  = np.random.randn(int(b/2),n)
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
    z2  = np.random.randn(int(b/2),n)
    #
    S21 = S11 * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * z2)
    # antithetic
    S22 = S12 * np.exp((r - delta - 0.5 * sigma ** 2) * dt 
           + sigma * math.sqrt(dt) * (-z2))
    S2  = np.concatenate((S21, S22))
    #    
    # HIGH ESTIMATOR -----------------------------------------------------------------------------
    #
    #
    # Time T-1
    #
    # calculating mesh (e' sbagliata l'implementazione del metodo antitetico, correggere!!!)
    #
    cont_2 = BS.BlackScholes('C', S2, K, r, delta, sigma, 1./3.) * math.exp(-r*(2./3.))
    exer_2 = np.maximum(S2 - K, 0)  * math.exp(-r*(2./3.))
    Q2     = np.maximum(exer_2, cont_2)
    #
    # control variate
    #
    call_2 = np.maximum(0, cont_2)
    #
    # calculating densities
    #
    density = np.zeros((b, b, n))
    for i in range(0, b):
        for j in range(0, b):
            c    = 1.0/(sigma*math.sqrt(dt)*S2[j,:])
            arg  = (np.log(S2[j,:]/S1[i,:]) - drift)/(sigma*math.sqrt(dt))
            density[i,j,:] = c*norm.pdf(arg)
    #
    # calculating weights
    #
    weights = np.zeros((b, b, n))
    for i in range(0, b):
        for j in range(0, b):
            somma = np.divide(np.sum(density[:,j,:], axis=0), b)
            weights[i, j,:] = np.divide(density[i,j,:], somma)
    #
    # Time T-2
    #
    Q1      = np.zeros((b,n))
    call_1  = np.zeros((b,n))
    for i in range(0, b):
        exer_2      = np.maximum(S1[i,:] - K, 0)  * math.exp(-r*dt)
        cont_2      = np.sum(weights[i,:,:] * Q2,     axis=0)/b
        call_1[i,:] = np.sum(weights[i,:,:] * call_2, axis=0)/b
        Q1[i,:]     = np.maximum(exer_2, cont_2)
    #
    # Time T = 0
    #
    cont_3      = np.sum(Q1, axis=0)/b
    est_high    = np.maximum(S0 - K, cont_3)
    #
    # control variate (non è questo, call_est è semplicemente la media di exer_2!!!!)
    #
    call_est    = np.sum(call_1, axis=0)/b
    #
    # LOW ESTIMATOR ------------------------------------------------------------------------------
    #
    # Time T = T - 2
    #
    final_payoff = BS.BlackScholes('C', S2, K, r, delta, sigma, 1./3.)
    est_low_2    = np.maximum(S2-K, final_payoff)*math.exp(-r*2./3.)
    
    xx           = range(0, b)
    weights_exe  = np.zeros((b, 2, n))
    weights_con  = np.zeros((b, b-2, n))
    Q_exe        = np.zeros((2,n))
    Q_con        = np.zeros((b-2,n))
    con          = np.zeros((b, int(b/2), n))
    exe          = np.zeros((b, int(b/2), n))
    for j in range(0,int(b/2)):
        y1      = np.asarray([j, j + int(b/2)])
        y2      = np.delete(xx, y1)
        Q_exe   = [est_low_2[l,:] for l in y1]
        Q_con   = [est_low_2[l,:] for l in y2] 
        for k in range(0,b):
            # These weights are used for calculating the estimator if the
            # choice is to continue
            somma1 = np.divide(np.sum(density[:,j,:], axis=0), b)
            weights_exe[k, 0, :] = np.divide(density[k,j , :], somma1)
            #
            somma2 = np.divide(np.sum(density[:,j + int(b/2),:], axis=0), b)
            weights_exe[k, 1, :] = np.divide(density[k,j + int(b/2), :], somma2)
            # These weights are used to check if to continue or not
            m = 0
            for i in y2:
                somma = np.divide(np.sum(density[:,i,:], axis=0), b)     
                weights_con[k, m, :] = np.divide(density[k, i, :], somma)
                m += 1
            
            con[k, j, :] = np.divide(np.sum(Q_con * weights_con[k, :, :], axis=0), (b-2))    
            exe[k, j, :] = np.divide(np.sum(Q_exe * weights_exe[k, :, :], axis=0), 2)
    #
    # Time T = T - 1
    #
    est         = np.zeros((b, int(b/2), n))       
    est_low_1   = np.zeros((b, n))
    for i in range(0, b):
        for k in range(0, int(b/2)):
            for l in range(0, n):
                if np.maximum(S1[i,l]-K,0)*math.exp(-r*dt) >= con[i, k, l]:
                    est[i, k, l] = np.maximum(S1[i,l]-K,0)*math.exp(-r*dt)
                else:
                    est[i, k, l] = exe[i, k, l]    
        est_low_1[i, :] = np.divide(np.sum(est[i], axis=0), (b/2))    
    
    somma   = np.divide(np.sum(est_low_1, axis=0), b)
    est_low = np.maximum(S0-K, somma)
    
    regr = linear_model.LinearRegression()
    regr.fit(est_high.reshape(-1,1), call_est)

    beta_1 = regr.coef_[0]
    r_sq = regr.score(est_high.reshape(-1,1), call_est)
    print('beta high : ', beta_1)
    #print('coefficient of determination high:', r_sq)
    
    regr.fit(est_low.reshape(-1,1), call_est)

    beta_2 = regr.coef_[0]
    r_sq = regr.score(est_low.reshape(-1,1), call_est)
    print('beta low  : ', beta_2)
    #print('coefficient of determination low:', r_sq)
    
    std_err_h      = 1.96 * np.std(est_high)/np.sqrt(n)
    std_err_l      = 1.96 * np.std(est_low)/np.sqrt(n)
    
    est_high       = np.average(est_high)
    est_low        = np.average(est_low)
    call_est       = np.average(call_est)

    est_high = est_high - beta_1 * (call_est - european_option) 
    est_low  = est_low  - beta_2 * (call_est - european_option) 
    
    point_estimate = 0.5 * est_low + 0.5 * est_high
    
    print('\n')
    print('monte carlo sim  = ' + str(n))
    print('branching        = ' + str(b))
    print('stock price      = ' + str(S0))

    print('high  estimator  = ' + str("%.3f" % est_high) + ' +/- ' + str("%.3f" % std_err_h)) 
    print('low   estimator  = ' + str("%.3f" % est_low)  + ' +/- ' + str("%.3f" % std_err_h))
    print('point estimator  = ' + str("%.3f" % point_estimate))
    print('black scholes    = ' + str("%.3f" % european_option))
    print('\n95% Confidence Interval')
    print(str("%.3f" % (est_low - std_err_l)) + ' - ' + str("%.3f" % (est_high + std_err_h)))
    print('\n')
    
    x.append(n)
    y.append(point_estimate)

if len(x) > 1:
    df = pd.DataFrame({'x':x, 'y':y})
    df.plot('x', 'y', kind='line')
    plt.hlines(np.average(y)  ,min(x), max(x), colors='red',   linewidth=2)
    plt.hlines(european_option,min(x), max(x), colors='green', linewidth=2)
    plt.show()

    filename = 'dump_stochastic_mesh.csv'
    df.to_csv(filename, index=False)

    print('average y = ' + str(np.average(y)))
    print('It worked!')