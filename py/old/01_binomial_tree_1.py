import numpy as np
from math import exp, sqrt
'''
 indispensabile creare una variabile di ambiente
 vedi
 https://stackoverflow.com/questions/54974444/pydev-eclipse-not-loading-mklinit-when-run-from-a-conda-environment
 definire PATH e PYTHONPATH
'''
def binomial_model(N, S0, sigma, r, K, Expiry):
    """
    N      = number of binomial iterations
    S0     = initial stock price
    sigma  = factor change of upstate
    r      = risk free interest rate per annum
    K      = strike price
    """
    delta_t     = Expiry/float(N)
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
    option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N] - K))
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            option[j, i] = (
                discount * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
    return [stock, option]


if __name__ == "__main__":
    print("Calculating example option price:")
    result = binomial_model(6, 100, 0.2, 0.2, 90, 1) #N, S0, sigma, r, K, Expiry
    print(result[0])