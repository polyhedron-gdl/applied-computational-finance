import math
import matplotlib
import pylab
import BS

from pylab import *
from matplotlib import pyplot as pl

np.random.seed(150000)
# Model Parameters
S0    =  70.0   # initial stock level
K     = 100.0   # strike price
T     =   1.00  # time-to-maturity
r     =   0.05  # short rate
sigma =   0.20  # volatility

# Simulation Parameters
b = 10000   # number of paths
M = 3      # number of points for each path
dt = T / M
df = math.exp(-r * dt)
#
# Random numbers generations. We use the function 'randn' which returns 
# a sample (or samples) from the "standard normal" distribution.
# If positive, int_like or int-convertible arguments are provided,
# 'randn' generates an array of shape (d0, d1, ..., dn), filled
# with random floats sampled from a univariate "normal" (Gaussian)
# distribution of mean 0 and variance 1
#
z1  = np.random.randn(M + 1, int(b/2))
#
# Stock Price Paths. The function 'cumsum' returns the cumulative sum 
# of the elements along a given axis, in this case we want to sum along
# the time axis (axis=0)
#
S     = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt 
       + sigma * math.sqrt(dt) * z1, axis=0))
S[0]  = S0

# plotting the first 1000 paths
t = np.linspace(0, T, M+1)
paths = S[:,1:100]
#
# plotting expiry price distribution. Note that the S matrix is build 
# with the first index running on the time slice and the second index
# running on the number of simulated path. Remember that Python programming 
# language supports negative indexing of arrays, something which is not 
# available in arrays in most other programming languages. This means that 
# the index value of -1 gives the last element, and -2 gives the second 
# last element of an array. The negative indexing starts from where the 
# array ends. So in this case we are selecting all the simulated value 
# (second index = ':') of the last simulated time (first index = '-1').
#
expiry = S[-1,:]
hist = np.histogram(expiry, 100)
index = np.arange(100)

pl.figure(figsize=(15,5))
pl.subplot(121)
pl.plot(t, paths)

pl.subplot(122)
pl.bar(index, hist[0])

pl.show()
#
# Asset value at T-1
S2 = S[-2,:]
payoff = np.maximum(K-expiry,0)*math.exp(-r *2*dt)

pl.figure()
pl.plot(S2, payoff,'.')
pl.show()

C = BS.BlackScholes('P',S2, K, r, sigma, 0.5*T)
pl.figure()
pl.plot(S2, payoff,'.')
pl.plot(S2, C, '.', color='r')
pl.show()



