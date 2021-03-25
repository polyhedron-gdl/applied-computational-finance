'''
Created on 5 feb 2021

@author: User
'''
import numpy as np

from numpy.random               import RandomState
from numpy.polynomial           import Polynomial
from matplotlib                 import pyplot as plt
from scipy.stats.distributions  import lognorm, rv_frozen
from pathlib                    import Path

class GeometricBrownianMotion:
    '''Geometric Brownian Motion.(with optional drift).'''
    def __init__(self, mu: float=0.0, sigma: float=1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) \
            -> np.array:
        assert t.ndim == 1, 'One dimensional time vector required'
        assert t.size > 0, 'At least one time point is required'
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), 'Increasing time vector required'
        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return np.exp(self.sigma * W.T + (self.mu - self.sigma**2 / 2) * t).T

    def distribution(self, t: float) -> rv_frozen:
        mu_t = (self.mu - self.sigma**2/2) * t
        sigma_t = self.sigma * np.sqrt(t)
        return lognorm(scale=np.exp(mu_t), s=sigma_t)
#_______________________________________________________________________________________________________________________________
#
def running_min_max(*array_seq):
    minimum, maximum = None, None
    for a in array_seq:
        cmin, cmax = a.min(), a.max()
        print(cmin, cmax)
        if minimum is None or cmin < minimum:
            minimum = cmin
        if maximum is None or cmax > maximum:
            maximum = cmax
    return minimum, maximum    
#_______________________________________________________________________________________________________________________________
#
if __name__ == "__main__":
    S0    = 680
    sigma = 0.2
    # zero interest rate so that we can ignore discounting
    gbm = GeometricBrownianMotion(mu=0.0, sigma=sigma)
    # time horizon is 5 years with a monthly frequency
    maturity  = 5
    frequency = 12
    t = np.linspace(0, maturity, frequency*maturity)
    
    nsim = 50
    rnd = RandomState(seed=1234)
    X = S0 * gbm.simulate(t, nsim, rnd)
    
    figsize = (10,6)
    
    plt.figure(figsize=figsize)
    plt.plot(t, X);
    plt.xlabel('Time t')
    plt.ylabel('Coin Value')
    
    #plt.show()

    strike = S0

    plt.figure(figsize=figsize)
    plt.plot(t, np.maximum(strike - X, 0));
    plt.xlabel('Time t')
    plt.ylabel('American Put Exercise Value')

    #plt.show()
    
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([4,5,6,7,8,9])
    z = np.array([6,7,8,9,10,11,12,13])
    
    (mins, maxs) = running_min_max(z, y, x)
    print((mins, maxs))