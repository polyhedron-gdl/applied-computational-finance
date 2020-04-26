import sys
import math
import numpy as np
import scipy.interpolate


S0    =  80.0   # initial stock level
K     = 100.0   # strike price
T     =   1.0   # time-to-maturity
r     =   0.01  # short rate
sigma =   0.20  # volatility
delta =   0.0   # dividend yield

payout = 'put'   # option type 
#
# Parameters to define the range in space and time
#
S_min  = 0.0;
S_max  = 200.0;
L      = S_max - S_min;

N      = 1000;                # Number of time steps
k      = float(T)/float(N);   # time step size
I      = 200;                 # Number of space steps
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
    A[i] = k * 0.5 * i * (sigma * sigma * i + r)
    B[i] = 1 + 1.0 * k * (sigma * sigma * i * i + r)
    C[i] = k * 0.5 * i * (sigma * sigma * i - r)
#
# Cycle on time
#
for n in range(1, N-1):
    print("Now running time step nr: " + str(n) + "/" + str(N-2))
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
            y       = (UOld[i] + A[i] * UNew[i + 1] + C[i] * UNew[i - 1]) / B[i]
            diff    = y - UNew[i]
            error   = error + diff * diff
            UNew[i] = UNew[i] + omega * diff
            UNew[i] = max(UNew[i], Payoff[i])
        m = m + 1
    #
    UOld[1:I-1] = UNew[1:I-1]

f = scipy.interpolate.interp1d(S,UNew)
opt_implicit = f(S0).item(0)

print('\r')
print('Option price (implicit finite diff)   : ', round(opt_implicit,3))