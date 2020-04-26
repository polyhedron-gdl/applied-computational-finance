import math
import scipy
import numpy             as np
import pandas            as pd
import scipy.stats       as ss
import matplotlib
import pylab

from pylab import *
from matplotlib import pyplot as pl
#
#-----------------------------------------------------------------------------------------------
#
#
# Black and Scholes
#
def d1(S0, K, r, delta, sigma, T):
    return (np.log(S0 / K) + (r - delta + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, delta, sigma, T):
    return (np.log(S0 / K) + (r - delta - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def BlackScholes(type, S0, K, r, delta, sigma, T):
    # for the cumulative normal distribution function we use norm.cdf from
    # the library scipy.stats
    if type=="C":
        return S0 * np.exp(- delta * T) * ss.norm.cdf(d1(S0, K, r, delta, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, delta, sigma, T))
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, delta, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, delta, sigma, T))
#
#-----------------------------------------------------------------------------------------------
#
def binomial_model_european(N, S0, sigma, r, K, Maturity):
    """
    N      = number of binomial iterations
    S0     = initial stock price
    sigma  = factor change of upstate
    r      = risk free interest rate per annum
    K      = strike price
    """
    delta_t     = Maturity/float(N)
    discount    = exp(-r*delta_t)
    #
    # u and d values are chosen according to the CRR model
    #
    u           = exp(sigma*sqrt(delta_t))
    d           = 1 / u
    p           = (exp(r*delta_t)- d) / (u - d)
    q           = 1 - p
    #
    # make stock price tree
    # 
    stock = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            stock[j, i] = S0 * (u ** (i - j)) * (d ** j)
    #
    # Initialize option matrix 
    #
    option = np.zeros([N + 1, N + 1])
    # 
    # Generate option prices recursively
    #
    #
    # We start from the maturity (the binomial tree is a backward 
    # in time algorithm remember?). At maturity we know the value 
    # of the option in all states, it is simply the payoff. In this
    # case the payoff is that of a put option.
    #
    # EXERCISE: Generalize this procedure by giving the user the 
    #           possibility to choose the option payoff parametrically
    #
    option[:, N] = np.maximum(np.zeros(N + 1), (K - stock[:, N]))
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            option[j, i] = (
                discount * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
    return [stock, option]
#
#-----------------------------------------------------------------------------------------------
#
def binomial_model_american(N, S0, sigma, r, K, Maturity):
    """
    N      = number of binomial iterations
    S0     = initial stock price
    sigma  = factor change of upstate
    r      = risk free interest rate per annum
    K      = strike price
    """
    delta_t     = Maturity/float(N)
    discount    = exp(-r*delta_t)
    u           = exp(sigma*sqrt(delta_t))
    d           = 1 / u
    p           = (exp(r*delta_t)- d) / (u - d)
    q           = 1 - p

    # make stock price tree
    stock = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            stock[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Generate option prices recursively
    option = np.zeros([N + 1, N + 1])
    option[:, N] = np.maximum(np.zeros(N + 1), (K - stock[:, N]))
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            option[j, i] = (
                discount * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
            #
            # dealing with early exercise
            #
            exercise     = np.maximum(0, K - stock[j, i])  
            option[j, i] = np.maximum(exercise, option[j, i])
    return [stock, option]
#
#-----------------------------------------------------------------------------------------------
#
def finite_difference_explicit(payout, S0, K, sigma, T, american = True):
    #
    # Parameters to define the range in space and time
    #
    S_min = 0.0;
    S_max = 200.0;
    L     = S_max - S_min;

    N     = 1000;               # Number of time steps
    k     = float(T)/float(N);  # time step size
    I     = 100;                # Number of space steps
    h     = float(L)/float(I);  # space step size

    sig   = sigma
    sig2  = sig*sig

    S     = np.zeros((I+1))
    u     = np.zeros((I+1,N+1))

    if payout == 'call':
        for i in range(0,I+1):
            S[i]   = i*h;
            u[i,0] = max(S[i]-K,0);
        #
        # We suppose that the range of S is sufficiently large, therefore impose the 
        # boundary conditions for all times:
        #
        # Value at the boundary 
        for n in range (0, N + 1):
            # If the underlying is zero even the option value is zero
            u[0,n] = 0.0;
            # for large values of the underlying value of the option tends asymptotically
            # to S - K * exp(-rt)
            u[I,n] = S[I] - K * math.exp(-r*k*n);

    if payout == 'put':
        for i in range(0,I+1):
            S[i] = S_min + i*h;
            u[i,0] = max(K - S[i],0);
        #
        # Value at the boundary 
        #
        for n in range (0, N + 1):
            u[0,n] = K * math.exp(-r*k*n);
            u[I,n] = 0.0;
    #
    # Implementation of the explicit method
    #
    for n in range(0,N): # time loop
        for i in range (0,I): # space loop
            A = 0.5 * (sig2 * i * i + r * i ) * k
            B = 1.0 - (sig2 * i * i + r) * k 
            C = 0.5 * (sig2 * i * i - r * i) * k
            u[i,n+1] = A * u[i+1,n] + B * u[i,n] + C * u[i-1,n];
            #
            # early exercise condition
            #
            if payout == 'put':
                exercise = np.maximum(K-S[i],0)
            else:
                exercise = np.maximum(S[i]-K,0)
            if american:    
                u[i, n+1] = np.maximum(exercise, u[i, n+1]) 
    
    f = scipy.interpolate.interp1d(S,u[:,N])
    return f(S0).item(0)               
#
#-----------------------------------------------------------------------------------------------
#
def finite_difference_implicit(payout, S0, K, sigma, T, american = True):
    import sys
    #
    # Parameters to define the range in space and time
    #
    S_min  =   0.0;
    S_max  = 150.0;
    L      = S_max - S_min;

    N      = 1002;                # Number of time steps
    k      = float(T)/float(N);   # time step size
    I      = 200;                  # Number of space steps
    h      = float(L)/float(I);   # space step size

    S      = np.zeros((I+1))
    A      = np.zeros((I+1))
    B      = np.zeros((I+1))
    C      = np.zeros((I+1))
    Payoff = np.zeros((I+1))
    UOld   = np.zeros((I+1))
    UNew   = np.zeros((I+1))
    #
    # Relaxation parameters
    #
    omega = 1.5
    error = 1e6
    #
    # Precision parameters
    #
    EPS     = 0.00001
    MAXITER = 100
    #
    # Initial Conditions
    #
    for i in range(0, I+1):
        S[i] = i * h
        if payout == 'call':
            UOld[i] = np.maximum(S[i] - K,0)
        else:
            UOld[i] = np.maximum(K - S[i],0)    
    Payoff = UOld
    #
    # Matrix elements
    #
    for i in range(1,I):
        A[i] = -k * 0.5 * i * (sigma * sigma * i - r)
        B[i] = 1 + 1.0 * k * (sigma * sigma * i * i + r)
        C[i] = -k * 0.5 * i * (sigma * sigma * i + r)
    #
    # Cycle on time
    #
    for n in range(1, N-1):
        sys.stdout.write("\r" + "Now running time step nr: " + str(n) + "/" + str(N-2))
        sys.stdout.flush()
        t = n * k
        #
        # initially UNew is set equal to UOld
        #
        UNew[1:I-1] = UOld[1:I-1]
        m = 0
        #
        # Boundary Conditions
        #
        if payout == 'call':
            UNew[0] = 0
            UNew[I] = h * I * math.exp(-delta * t) - K * math.exp(-r * t)
        else:
            UNew[0] = K * math.exp(-r * t)
            UNew[I] = 0

        while (error > EPS) and (m < MAXITER): 
            error = 0
            for i in range(1,I):
                y       = (UOld[i] - A[i] * UNew[i - 1] - C[i] * UNew[i + 1] ) / B[i]
                diff    = y - UNew[i]
                error   = error + diff * diff
                UNew[i] = UNew[i] + omega * diff
                #
                # this is were we introduce the early exercise opportunity
                #
                if american:
                    UNew[i] = max(UNew[i], Payoff[i])
            m = m + 1
        #
        UOld[1:I-1] = UNew[1:I-1]

    print('\r')
    f = scipy.interpolate.interp1d(S,UNew)
    return f(S0).item(0)
#
#-----------------------------------------------------------------------------------------------
#
if __name__ == "__main__": 
    
    S0    =  80.0   # initial stock level
    K     = 100.0   # strike price
    T     =   1.0   # time-to-maturity
    r     =   0.01  # short rate
    sigma =   0.20  # volatility
    delta =   0.0   # dividend yield    
    N     = 100
    
    opt_european   = BlackScholes('P', S0, K, r, 0, sigma, T)
    opt_binomial_e = binomial_model_european(N, S0, sigma, r, K, T)[1][0][0] 
    opt_binomial_a = binomial_model_american(N, S0, sigma, r, K, T)[1][0][0]
    opt_explicit   = finite_difference_explicit('put', S0, K, sigma, T, True)
    opt_implicit   = finite_difference_implicit('put', S0, K, sigma, T, True)
    
    print('Option price (analytic)               : ', round(opt_european,3))
    print('Option price (binomial tree european) : ', round(opt_binomial_e,3))
    print('Option price (binomial tree american) : ', round(opt_binomial_a,3))
    print('Option price (explicit finite diff)   : ', round(opt_explicit,3))
    print('Option price (implicit finite diff)   : ', round(opt_implicit,3))            