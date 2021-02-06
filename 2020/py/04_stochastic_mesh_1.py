# Note:
#
# - utilizzare il fancy indexing per la selezione dei valori
#   da utilizzare per exercise e continuation
#
# - questa versione cerca di correggere un presunto errore nell'indicizzazione
#   del vettore dei pesi nella parte dello stimatore low bias
#
# - controllare l'implementazione del metodo delle variabili antitetiche, sembra
#   sbagliata
#
import math
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import BS

from scipy.stats import norm 

# np.random.seed(150000)
# Option Parameters
K     = 100.0   # strike price
delta =   0.0  # dividend yield 
T     =   1.00  # time-to-maturity
r     =   0.00  # short rate

# Asset Parameters
S0    = 120.0   # initial stock level
sigma =   0.20  # volatility

# Simulation Parameters
b  = 30                     # branching parameter (must be an even number!!!)
n  = 10                     # Monte Carlo reiterations
t  = [0.,1./3.,2./3.,1.]    # exercise opportunities
dt = t[1]-t[0]
df = math.exp(-r * dt)

drift = (r - delta - .5 * sigma**2) * dt

x = []
y = []

est_high        = 0
est_low         = 0
point_estimate  = 0

stock_price     = []
high_estimator  = []
low_estimator   = []
point_estimator = []
true_value      = []
SEh             = []
SEl             = []
AbsErr          = []
confidence_int  = []

#for S0 in range(120, 130, 10):
for n in range(10000, 10002, 2):
    european_option = BS.BlackScholes('C', S0, K, r,delta,sigma, T)
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
    # Time t2 = T - 1
    #
    # calculating mesh
    #
    cont_2     = BS.BlackScholes('C', S2, K, r, delta, sigma, 1./3.) * math.exp(-r*(2./3.))
    exer_2     = np.maximum(S2 - K, 0)  * math.exp(-r*(2./3.))
    est_high_2 = np.maximum(exer_2, cont_2)
    #
    # calculating densities
    #
    density = np.zeros((b, b, n))
    for i in range(0, b):
        for j in range(0, b):
            c    = 1.0/(sigma * math.sqrt(dt) * S2[i,:])
            arg  = (np.log(S2[i,:]/S1[j,:]) - drift)/(sigma * math.sqrt(dt))
            density[j,i,:] = c * norm.pdf(arg)
    #
    # calculating weights
    #
    weights = np.zeros((b, b, n))
    for i in range(0, b):
        for j in range(0, b):
            somma = np.divide(np.sum(density[:,j,:], axis=0), b)
            weights[i, j,:] = np.divide(density[i, j,:], somma)
    #
    # Time t1 = T - 2
    #
    #
    est_high_1      = np.zeros((b,n))
    for i in range(0, b):
        exer_1          = np.maximum(S1[i,:] - K, 0)  * math.exp(-r*dt)
        cont_1          = np.divide(np.sum(weights[i,:,:] * est_high_2 , axis=0), b)
        est_high_1[i,:] = np.maximum(exer_1, cont_1)
    #
    # Time T = 0
    #
    cont_3      = np.divide(np.sum(est_high_1, axis=0), b)
    est_high    = np.maximum(S0 - K, cont_3)
    #
    # LOW ESTIMATOR ------------------------------------------------------------------------------
    #
    # Time t2 = T - 1
    #
    final_payoff = BS.BlackScholes('C', S2, K, r,delta,sigma, 1./3.)
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
        # we remove from xx values selected by y1
        y2      = np.delete(xx, y1)
        Q_exe   = [est_low_2[l,:] for l in y1]
        Q_con   = [est_low_2[l,:] for l in y2] 
        for k in range(0,b):
            # These weights are used for calculating the estimator if the
            # choice is to continue
            somma1 = np.divide(np.sum(density[:,j,:], axis=0), b)
            weights_exe[k, 0, :] = np.divide(density[k, j , :], somma1)
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
    # Time t1 = T - 2
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

    std_err_h      = 1.96 * np.std(est_high)/np.sqrt(n)
    std_err_l      = 1.96 * np.std(est_low)/np.sqrt(n)
    
    est_high       = np.average(est_high)
    est_low        = np.average(est_low)

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

    stock_price.append(S0)
    high_estimator.append(est_high)
    low_estimator.append(est_low)
    point_estimator.append(point_estimate)
    true_value.append(european_option)
    SEh.append(std_err_h)
    SEl.append(std_err_l)
    AbsErr.append(abs(point_estimate-european_option))
    confidence_int.append(str("%.3f" % (est_low - std_err_l)) + ' - ' + str("%.3f" % (est_high + std_err_h)))
    
    x.append(n)
    y.append(point_estimate)


header = ['stock price','high est','low est', 'point est','true value','Abs Err', 'SE high','SE low','95% conf interval']
data   = {'stock price':stock_price,
          'high est':high_estimator,
          'low est':low_estimator, 
          'point est':point_estimator,
          'true value':true_value,
          'Abs Err':AbsErr,
          'SE high':SEh,
          'SE low':SEl,
          '95% conf interval':confidence_int}
frame  = pd.DataFrame(data,columns=header)
frame.to_csv('dump_stochastic_mesh_0.csv', index=False, sep=';')
print(frame)

if len(x) >= 1000:
    df = pd.DataFrame({'x':x, 'y':y})
    df.plot('x', 'y', kind='line')
    df.to_csv('dump_stochastic_mesh_plot_1.csv', index=False, sep =';') 
    plt.hlines(np.average(y)  ,min(x), max(x), colors='green',   linewidth=2)
    plt.hlines(european_option,min(x), max(x), colors='red', linewidth=2)
    plt.show()


    print('average y = ' + str(np.average(y)))
    print('It worked!')