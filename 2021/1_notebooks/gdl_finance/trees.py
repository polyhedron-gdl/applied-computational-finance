'''
Created on 24 apr 2020

@author: giovanni della lunga
'''
from matplotlib import pyplot    as plt    
import numpy                     as np
#
#------------------------------------------------------------------------------------- 
#
def binomial_model_european(N, S0, sigma, r, K, Maturity, opt_type = 'put'):
    """
    N      = number of binomial levels
    S0     = initial stock price
    sigma  = factor change of upstate
    r      = risk free interest rate per annum
    K      = strike price
    """
    if S0 < 0 or N < 0 or sigma < 0 or K < 0 or Maturity < 0 or opt_type not in ['call','put']: 
        raise ValueError('Negative value in input!')
    if type(N) not in [int]:
        raise TypeError('The number of levels must be an non-negative integer number!')

    try:
        delta_t     = Maturity/float(N)
        discount    = np.exp(-r*delta_t)
        #
        # u and d values are chosen according to the CRR model
        #
        u = np.exp(sigma*np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(r*delta_t)- d) / (u - d)
        q = 1 - p
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
        if opt_type == 'put':
            option[:, N] = np.maximum(np.zeros(N + 1), (K - stock[:, N]))
        else:
            option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N]) - K)
            
        for i in range(N - 1, -1, -1):
            for j in range(0, i + 1):
                option[j, i] = (
                    discount * (p * option[j, i + 1] + q * option[j + 1, i + 1])
                )
        return [stock, option]
    except:
        return []
#
#------------------------------------------------------------------------------------- 
#
def random_tree(b, labels = [], nlevel = 2):
    if b < 10:
        plt.figure(figsize=[10, 10])
        # 
        # calculating tree points coordinates
        #
        x       = (b**nlevel)*[nlevel]
        y       = list(range(b**nlevel))
        y_last  = y
        for i in range(nlevel-1,-1,-1):
            k     = b**i
            slide = []
            for j in range(k):
                new_point = np.average(y_last[j*b:j*b+b])
                x.append(i)
                y.append(new_point)
                slide.append(new_point)
            y_last = slide
        x = x[::-1]
        y = y[::-1]  
        #
        # setting labels
        #
        if labels:
            for k in range(len(x)):
                plt.text(x[k]-0.1,y[k]+0.1,labels[k])    
        #
        # building graph
        #
        imax  = 0
        x_old = [x[0]]
        y_old = [y[0]]
        for i in range(1, nlevel+1):
            imax = imax + b**i 
            imin = imax - b**i + 1
            x_new = x[imin:imax+1]
            y_new = y[imin:imax+1]
            for k in range(len(x_old)):
                x_plt = []
                y_plt = []
                for j in range(k*b,k*b+b):
                    x_plt.append(x_old[k])
                    x_plt.append(x_new[j])
                    y_plt.append(y_old[k])
                    y_plt.append(y_new[j])
                    plt.plot(np.array(x_plt), np.array(y_plt), 'bo-', color='blue')
                        
            x_old = x_new
            y_old = y_new

        plt.grid(True)
        plt.show()
#
#------------------------------------------------------------------------------------- 
#
def binomial_model_american(N, S0, sigma, r, K, Maturity, opt_type = 'put'):
    """
    N      = number of binomial iterations
    S0     = initial stock price
    sigma  = factor change of upstate
    r      = risk free interest rate per annum
    K      = strike price
    """
    delta_t     = Maturity/float(N)
    discount    = np.exp(-r*delta_t)
    u           = np.exp(sigma*np.sqrt(delta_t))
    d           = 1 / u
    p           = (np.exp(r*delta_t)- d) / (u - d)
    q           = 1 - p

    # make stock price tree
    stock = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            stock[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Generate option prices recursively
    option = np.zeros([N + 1, N + 1])
    if opt_type  == 'put':
        option[:, N] = np.maximum(np.zeros(N + 1), (K - stock[:, N]))
    else:
        option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N]) - K)
        
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            option[j, i] = (
                discount * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
            #
            # dealing with early exercise
            #
            if opt_type == 'put':
                exercise     = np.maximum(0, K - stock[j, i])  
            else:
                exercise     = np.maximum(0, stock[j, i] - K)
                
            option[j, i] = np.maximum(exercise, option[j, i])

    return [stock, option]
#
#------------------------------------------------------------------------------------- 
#
def binomial_tree(n, labels = []):

    plt.figure(figsize=[10, 10])

    for i in range(n):
        x = [1, 0, 1]
        for j in range(i):
            x.append(0)
            x.append(1)
        x = np.array(x) + i
        y = np.arange(-(i+1), i+2)[::-1]
        plt.plot(x, y, 'bo-')
    #
    # find point indexing
    #
    #for k in range(len(x)):
    #    plt.text(x[k]-0.1,y[k]+0.1,labels[k])    
    #
    # put labels
    # 
    if labels:
        k = 0
        for xx in range(0, n+1):
            for yy in range(xx,-xx-1, -2):
                plt.text(xx - 0.1,yy + 0.2,labels[k])  
                k = k+1

    plt.xticks(range(n+1))
    plt.grid(True)
    plt.show() 
#
#------------------------------------------------------------------------------------- 
#
def trinomial_tree(n, labels = []):
    t_par = {'branching' : 3, 'levels' : n, 'type' : 'recombining', 'labels' : labels}
    general_tree(t_par)    
#
#------------------------------------------------------------------------------------- 
#
def mesh(n, b):
    fig = plt.figure(figsize=[15, 10])
    #
    # Tree properties
    #
    # n number of levels
    # b branching parameter
    # 
    # calculating tree points coordinates
    #
    x       = [0]
    y       = [float((b-1))/2.0]
    for i in range(1,n+1):
        for j in range(b):
            x.append(i)
            y.append(j)
    #
    # building trajectories
    #
    fig.add_subplot(121)
    
    x_old = b*[x[0]]
    y_old = b*[y[0]]
    for i in range(1, n + 1):
        imax = b*i 
        imin = imax - b + 1
        x_new = x[imin:imax+1]
        y_new = y[imin:imax+1]
        for j in range(len(x_new)):
            x_plt = []
            y_plt = []
            x_plt.append(x_old[j])
            x_plt.append(x_new[j])
            y_plt.append(y_old[j])
            y_plt.append(y_new[j])
            plt.plot(x_plt, y_plt, 'bo-')
 
        x_old = x_new
        y_old = y_new
    
    plt.title('Fig. A - Building Trajectories',     fontsize=12)
    plt.xlabel('Time Step (index i)',        fontsize=12)
    plt.ylabel('Simulation Path (index j)',  fontsize=12)
    #
    # building interactions
    #
    fig.add_subplot(122)
    
    x_old = [x[0]]
    y_old = [y[0]]
    for i in range(1, n+1):
        imax = b*i 
        imin = imax - b + 1
        x_new = x[imin:imax+1]
        y_new = y[imin:imax+1]
        x_plt = []
        y_plt = []
        for k in range(len(x_old)):
            for j in range(len(x_new)):
                x_plt.append(x_old[k])
                x_plt.append(x_new[j])
                y_plt.append(y_old[k])
                y_plt.append(y_new[j])
        
        x_old = x_new
        y_old = y_new
        plt.plot(x_plt, y_plt, 'bo-')
    
    plt.title('Fig. B - Building Interactions',     fontsize=12)
    plt.xlabel('Time Step (index i)',        fontsize=12)
    plt.ylabel('Simulation Path (index j)',  fontsize=12)
    
    plt.grid(True)
    plt.show()
#
#------------------------------------------------------------------------------------- 
#
def npoints_per_level(t_type, branching, level):
    if t_type == 'recombining':
        return branching*level-level+1
    else:
        return branching**level
#
#------------------------------------------------------------------------------------- 
#
def general_tree(t_par):
    
    b         = t_par['branching']
    nlevel    = t_par['levels']
    tree_type = t_par['type']
    labels    = t_par['labels']
    step      = 1
    if tree_type != 'recombining': step = b
    
    if b < 10:
        plt.figure(figsize=[10, 8])
        # 
        # calculating tree points coordinates
        #
        if tree_type == 'recombining':
            x       = (b*nlevel-nlevel +1)*[nlevel]
            y       = list(range(b*nlevel - nlevel+1))
            delta   = (y[b-1]-y[0])/2.0
            for i in range(nlevel-1,-1,-1):
                k     = b*i - i + 1 
                for j in range(k):
                    x.append(i)
                    y.append(j + (nlevel-i)*delta)

            x = x[::-1]
            y = y[::-1]  
        else:    
            x       = (b**nlevel)*[nlevel]
            y       = list(range(b**nlevel))
            y_last  = y
            for i in range(nlevel-1,-1,-1):
                k     = b**i
                slide = []
                for j in range(k):
                    new_point = np.average(y_last[j*b:j*b+b])
                    x.append(i)
                    y.append(new_point)
                    slide.append(new_point)
                y_last = slide
            x = x[::-1]
            y = y[::-1]  
        
        #
        # setting labels
        #
        if labels:
            for k in range(len(x)):
                plt.text(x[k]-0.1,y[k]+0.1,labels[k])    
        #
        # plot graph
        #
        imax  = 0
        x_old = [x[0]]
        y_old = [y[0]]
        for i in range(1, nlevel+1):
            npoints = npoints_per_level(tree_type, b, i)
            imax = imax + npoints 
            imin = imax - npoints + 1
            x_new = x[imin:imax+1]
            y_new = y[imin:imax+1]
            for k in range(len(x_old)):
                x_plt = []
                y_plt = []
                for j in range(k*step,k*step+b):
                    x_plt.append(x_old[k])
                    x_plt.append(x_new[j])
                    y_plt.append(y_old[k])
                    y_plt.append(y_new[j])
                    plt.plot(np.array(x_plt), np.array(y_plt), 'bo-', color='blue')
                            
            x_old = x_new
            y_old = y_new

        plt.grid(True)
        plt.show()
#
#------------------------------------------------------------------------------------- 
#
if __name__ == "__main__":
    # test #
    labels = 200*['label']
    t_par = {'branching' : 2, 'levels' : 8, 'type' : 'recombining', 'labels' : labels}
    #general_tree(t_par)    
    #binomial_tree(3, [])   
    trinomial_tree(3, labels)
    #mesh(3,3)