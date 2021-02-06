import numpy as np
import scipy.stats as ss

#Black and Scholes
def d1(S0, K, r, delta, sigma, T):
    return (np.log(S0/K) + (r - delta + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, delta, sigma, T):
    return (np.log(S0 / K) + (r - delta - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def BlackScholes(opt_type, S0, K, r, delta, sigma, T):
    if opt_type=="C":
        return S0 * np.exp(-delta * T) * ss.norm.cdf(d1(S0, K, r, delta, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, delta, sigma, T))
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, delta, sigma, T)) - S0 * np.exp(-delta*T) * ss.norm.cdf(-d1(S0, K, r, delta, sigma, T))

def AsianGeometric(opt_type, S0, K, r, delta, sigma, T, m_ave):
    m_ave = float(m_ave)
    T     = float(T)
    h  = T / m_ave
    mg = np.log(S0) + (r - sigma**2 / 2.0) * (T + h) / 2.0
    sg = np.sqrt(sigma**2 * h * (2.0 * m_ave + 1.0) * (m_ave + 1.0) / (6.0 * m_ave))

    d1 = (mg - np.log(K) + sg**2) / sg
    d2 = d1 - sg
    if opt_type=='C':
        return np.exp(-r * T) * (np.exp(mg + sg**2 / 2.0) * ss.norm.cdf(d1) - K * ss.norm.cdf(d2))
    else:
        return -1
    
    
if __name__ == "__main__":
    # Financial Parameters
    S0    =  90.0   # initial stock level
    K     = 100.0   # strike price
    T     =   1.0   # time-to-maturity
    r     =   0.05  # short rate
    sigma =   0.20  # volatility
    delta =   0.0   # dividend yield
    
    C = BlackScholes('C', S0, K, r, delta, sigma, T)
    P = BlackScholes('P', S0, K, r, delta, sigma, T)
    
    check = (C - P == S0 - K * np.exp(-r*T))
    
    print(C-P, S0-K*np.exp(-r*T))
    print(check)