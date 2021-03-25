'''
Created on 31 gen 2021

@author: User
'''
import time
import math
import pandas as pd
import numpy  as np
import scipy.interpolate

def HestonExplicitPDE_1_0(Si, vi, p_heston, p_option, p_grid):
    '''
    ' Finite differences for the Heston PDE for a European Call
    ' Reference:
    ' In 'T Hout and Foulon "ADI Finite Difference Schemes for Option Pricing in the Heston Model with Correlation" 
    ' Int J of Num Analysis and Modeling, 2010.
    ' INPUTS
    '    params     = 6x1 vector of Heston parameters
    '    Strike     = Strike price
    '    r          = risk free rate
    '    q          = Dividend yield
    '    Smax, Smin = Max and Min values of stock price
    '    Vmax, Vmin = Max and Min values of volatility
    '    Tmax, Tmin = Max and Min values of maturity
    '    NS, NV, NT = Number of points on stock price, volatility, and maturity grids
    '    GridType   = Type of Grid ("Uniform" or "NonUniform")
    '    Si         = Value of Spot price at which to interpolate on U(S,V)
    '    Vi         = Value of Volatility at which to interpolate on U(S,V)
    ' OUTPUT
    '    U(Si,Vi)   = Interpolated value of U(S,V) and points (Si,Vi)
    '''
    start = time.time()
    print('program starts...')
    #
    # Number of Steps in Each Dimension
    #
    NS = p_grid['NS']
    NV = p_grid['NV']
    NT = p_grid['NT']
    #
    # Option Features
    #
    Strike = p_option['Strike']
    r      = p_option['r']
    q      = p_option['q']
    #
    # Boundary Values for Stock Price, Volatility, and Maturity
    #
    Smin = p_grid['Smin']
    Smax = p_grid['Smax']
    Vmin = p_grid['Vmin'] 
    Vmax = p_grid['Vmax']
    Tmin = p_grid['Tmin']
    Tmax = p_grid['Tmax']
    # Heston parameters. Note: only kappa, theta, and sigma are needed
    kappa  = p_heston['kappa']
    theta  = p_heston['theta']
    sigma  = p_heston['sigma']
    sigma2 = sigma*sigma
    # Increment for Stock Price, Volatility, and Maturity
    ds = (Smax - Smin) / (NS - 1)
    dv = (Vmax - Vmin) / (NV - 1)
    dt = (Tmax - Tmin) / (NT - 1)
    # Building a Uniform Grid
    Mat  = np.arange(Tmin, Tmax + dt, dt)
    Spot = np.arange(Smin, Smax + ds, ds)
    Vol  = np.arange(Vmin, Vmax + dv, dv)  
    # Make sure the array have the right dimension
    Mat  = Mat[:NT]
    Spot = Spot[:NS]
    Vol  = Vol[:NV]
    
    # Initialize the 2-D grid with zeros
    U = np.zeros((NS, NV))
    
    # Temporary grid for previous time steps
    u = np.zeros((NS, NV))
    
    # Boundary condition for Call Option at t = Maturity
    
    for j in range(NV):
        U[:,j] = np.maximum(Spot - Strike, 0)
    
    # loop on maturity
    c1 = (1 - r * dt - kappa * theta * dt / dv)
    c2 = dt * 0.5 * (r - q)/ds
    c3 = kappa * theta * (dt/dv)
    for tt in range(NT):
        # Boundary condition for Smin and Smax
        U[0,:]    = 0
        U[NS-1,:] = np.max(Smax - Strike, 0)
        # Boundary condition for Vmax
        U[:,NV-1] = np.maximum(Spot - Strike, 0)
        # Update the temporary grid u(s,t) with the boundary conditions
        u = U
        # Boundary condition for Vmin.
        # Previous time step values are in the temporary grid u(s,t)
           
        #U[1:NS-2,0] = u[1:NS-2,0] * (1 - r * dt - kappa * theta * dt / dv) \
        #+ dt * 0.5 * (r - q) * Spot[1:NS-2] * (u[2:NS-1, 0] - u[0:NS-3,0]) / ds \
        #+ kappa * theta * (dt/dv) * u[1:NS-2,1]  
        
        for ss in range(1, NS-1):
            U[ss, 0] = c1 * u[ss, 0] \
                     + c2 * (ss+1) * (u[ss + 1, 0] - u[ss - 1, 0]) \
                     + c3 * u[ss, 1]
        
        # Update the temporary grid u(s,t) with the boundary conditions
        u = U
        # Interior points of the grid (non boundary).
        # As usual previous time step values are in the temporary grid u(s,t)
        for i in range(1, NS - 1):
            for j in range(1, NV - 1):
                C1 = (1 - dt * i * i * j * dv - sigma2 * j * dt / dv - r * dt)
                C2 = (0.5 * dt * i * i * j * dv - 0.5 * dt * (r - q) * i)
                C3 = (0.5 * dt * i * i * j * dv + 0.5 * dt * (r - q) * i)
                C4 = (0.5 * dt * sigma2 * j / dv - 0.5 * dt * kappa * (theta - j * dv) / dv)
                C5 = (0.5 * dt * sigma2 * j / dv + 0.5 * dt * kappa * (theta - j * dv) / dv)
                C6 = 0.25 * dt * sigma * i * j
                # The PDE
                U[i, j] = C1 * u[i, j] + C2 * u[i - 1, j] + C3 * u[i + 1, j] \
                       + C4 * u[i, j - 1] + C5 * u[i, j + 1] \
                       + C6 * (u[i + 1, j + 1] + u[i - 1, j - 1] - u[i - 1, j + 1] - u[i + 1, j - 1])
        
    #df = pd.DataFrame(U, index = Spot.round(4), columns=Vol.round(4))
    #df = df.round(4)
    #df.to_csv('out.txt', sep=';')
                      
    U = U.transpose()        
        
    f = scipy.interpolate.interp2d(Spot, Vol, U)
    end = time.time()
    print('program ends...')
    return (f(Si, vi), end-start)    
#_______________________________________________________________________________________________________________________________
#
def HestonExplicitPDE_1_1(Si, vi, p_heston, p_option, p_grid):
    '''
    ' Finite differences for the Heston PDE for a European Call
    ' Reference:
    ' In 'T Hout and Foulon "ADI Finite Difference Schemes for Option Pricing in the Heston Model with Correlation" 
    ' Int J of Num Analysis and Modeling, 2010.
    ' INPUTS
    '    params     = 6x1 vector of Heston parameters
    '    Strike     = Strike price
    '    r          = risk free rate
    '    q          = Dividend yield
    '    Smax, Smin = Max and Min values of stock price
    '    Vmax, Vmin = Max and Min values of volatility
    '    Tmax, Tmin = Max and Min values of maturity
    '    NS, NV, NT = Number of points on stock price, volatility, and maturity grids
    '    GridType   = Type of Grid ("Uniform" or "NonUniform")
    '    Si         = Value of Spot price at which to interpolate on U(S,V)
    '    Vi         = Value of Volatility at which to interpolate on U(S,V)
    ' OUTPUT
    '    U(Si,Vi)   = Interpolated value of U(S,V) and points (Si,Vi)
    '''
    start = time.time()
    print('program starts...')
    #
    # Number of Steps in Each Dimension
    #
    NS = p_grid['NS']
    NV = p_grid['NV']
    NT = p_grid['NT']
    #
    # Option Features
    #
    Strike = p_option['Strike']
    r      = p_option['r']
    q      = p_option['q']
    #
    # Boundary Values for Stock Price, Volatility, and Maturity
    #
    Smin = p_grid['Smin']
    Smax = p_grid['Smax']
    Vmin = p_grid['Vmin'] 
    Vmax = p_grid['Vmax']
    Tmin = p_grid['Tmin']
    Tmax = p_grid['Tmax']
    # Heston parameters. Note: only kappa, theta, and sigma are needed
    kappa = p_heston['kappa']
    theta = p_heston['theta']
    sigma = p_heston['sigma']
    # Increment for Stock Price, Volatility, and Maturity
    ds = (Smax - Smin) / (NS - 1)
    dv = (Vmax - Vmin) / (NV - 1)
    dt = (Tmax - Tmin) / (NT - 1)
    # Building a Uniform Grid
    Mat  = np.arange(Tmin, Tmax + dt, dt)
    Spot = np.arange(Smin, Smax + ds, ds)
    Vol  = np.arange(Vmin, Vmax + dv, dv)  
    # Make sure the array have the right dimension
    Mat  = Mat[:NT]
    Spot = Spot[:NS]
    Vol  = Vol[:NV]
    
    # Initialize the 2-D grid with zeros
    U = np.zeros((NS, NV))
    
    # Temporary grid for previous time steps
    u = np.zeros((NS, NV))
    
    # Boundary condition for Call Option at t = Maturity
    for j in range(NV):
        U[:,j] = np.maximum(Spot - Strike, 0)
    
    # loop on maturity
    c1 = (1 - r * dt - kappa * theta * dt / dv)
    c2 = dt * 0.5 * (r - q)/ds
    c3 = kappa * theta * (dt/dv)
    for tt in range(NT):
        # Boundary condition for Smin and Smax
        U[0,:]    = 0
        U[NS-1,:] = np.max(Smax - Strike, 0)
        # Boundary condition for Vmax
        U[:,NV-1] = np.maximum(Spot - Strike, 0)
        # Update the temporary grid u(s,t) with the boundary conditions
        u = U
        # Boundary condition for Vmin.
        # Previous time step values are in the temporary grid u(s,t)
           
        U[1:NS-2,0] = c1 * u[1:NS-2,0] \
                    + c2 * Spot[1:NS-2] * (u[2:NS-1, 0] - u[0:NS-3,0]) \
                    + c3 * u[1:NS-2,1]  
        
        # Update the temporary grid u(s,t) with the boundary conditions
        u = U
        # Interior points of the grid (non boundary).
        # As usual previous time step values are in the temporary grid u(s,t)
        for i in range(1, NS - 1):
            for j in range(1, NV - 1):
                C1 = (1 - dt * i * i * j * dv - sigma * sigma * j * dt / dv - r * dt)
                C2 = (0.5 * dt * i * i * j * dv - 0.5 * dt * (r - q) * i)
                C3 = (0.5 * dt * i * i * j * dv + 0.5 * dt * (r - q) * i)
                C4 = (0.5 * dt * sigma * sigma * j / dv - 0.5 * dt * kappa * (theta - j * dv) / dv)
                C5 = (0.5 * dt * sigma * sigma * j / dv + 0.5 * dt * kappa * (theta - j * dv) / dv)
                C6 = 0.25 * dt * sigma * i * j
                # The PDE
                U[i, j] = C1 * u[i, j] + C2 * u[i - 1, j] + C3 * u[i + 1, j] \
                       + C4 * u[i, j - 1] + C5 * u[i, j + 1] \
                       + C6 * (u[i + 1, j + 1] + u[i - 1, j - 1] - u[i - 1, j + 1] - u[i + 1, j - 1])
        
    #df = pd.DataFrame(U, index = Spot.round(4), columns=Vol.round(4))
    #df = df.round(4)
    #df.to_csv('out.txt', sep=';')
                      
    U = U.transpose()        
        
    f = scipy.interpolate.interp2d(Spot, Vol, U)
    end = time.time()
    print('program ends...')
    return (f(Si, vi), end - start)    
#_______________________________________________________________________________________________________________________________
#
def BuildGrid(grid_type, p_grid, p_option):
    
    K     = p_option['Strike']
    Smax  = p_grid['Smax']
    Smin  = p_grid['Smin']
    Vmax  = p_grid['Vmax']
    Vmin  = p_grid['Vmin']
    NS    = p_grid['NS']
    NV    = p_grid['NV']
    
    if grid_type == 'uniform':
        # Increment for Stock Price, Volatility, and Maturity
        ds = (Smax - Smin) / (NS - 1)
        dv = (Vmax - Vmin) / (NV - 1)
        # Building a Uniform Grid
        Spot = np.arange(Smin, Smax + ds, ds)
        Vol  = np.arange(Vmin, Vmax + dv, dv)  
        # Make sure the array have the right dimension
        Spot = Spot[:NS]
        Vol  = Vol[:NV]
    else:
        C = K / 5
        dz = 1 / (NS - 1) * (math.asinh((Smax - K) / C) - math.asinh(-K / C))
        # The Spot Price Grid
        Z    = np.zeros(NS)
        Spot = np.zeros(NS)
        for i in range(NS):
            Z[i] = math.asinh(-K / C) + (i - 1) * dz
            Spot[i] = K + C * math.sinh(Z[i])
        
        # The volatility grid
        d = Vmax / 10
        dn = math.asinh(Vmax / d) / (NV - 1)
        N = np.zeros(NV)
        Vol = np.zeros(NV)
        for j in range(NV):
            N[j] = (j - 1) * dn
            Vol[j] = d * math.sinh(N[j])

    return (Spot, Vol)    
#_______________________________________________________________________________________________________________________________
#
def HestonExplicitPDE_NU_1_1(Si, vi, p_heston, p_option, p_grid):
    
    start = time.time()
    print('program starts...')
    #
    # Number of Steps in Each Dimension
    #
    NS = p_grid['NS']
    NV = p_grid['NV']
    NT = p_grid['NT']
    #
    # Option Features
    #
    Strike = p_option['Strike']
    r      = p_option['r']
    q      = p_option['q']
    #
    # Boundary Values for Stock Price, Volatility, and Maturity
    #
    Smax = p_grid['Smax']
    Tmin = p_grid['Tmin']
    Tmax = p_grid['Tmax']
    #
    # Heston parameters. Note: only kappa, theta, and sigma are needed
    #
    kappa = p_heston['kappa']
    theta = p_heston['theta']
    sigma = p_heston['sigma']
    rho   = p_heston['rho']
    #
    # Building grid
    #
    (Spot, Vol) = BuildGrid('non-uniform', p_grid, p_option)
    
    # Increment for Maturity
    dt = (Tmax - Tmin) / (NT - 1)
    
    # Initialize the 2-D grid with zeros
    U = np.zeros((NS, NV))
    
    # Temporary grid for previous time steps
    u = np.zeros((NS, NV))

    # Boundary condition for Call Option at t = Maturity
    for j in range(NV):
        U[:,j] = np.maximum(Spot - Strike, 0)

    for tt in range(NT):
        # Boundary condition for Smin and Smax
        U[0,:]    = 0
        U[NS-1,:] = np.max(Smax - Strike, 0)
        # Boundary condition for Vmax
        U[:,NV-1] = np.maximum(Spot - Strike, 0)
        # Update the temporary grid u(s,t) with the boundary conditions
        u = U
        # Boundary condition for Vmin.
        # Previous time step values are in the temporary grid u(s,t)
        for ss in range(1, NS-1):
            DerV     = (u[ss, 1] - u[ss, 0]) / (Vol[1] - Vol[0])
            DerS     = (u[ss + 1, 0] - u[ss - 1, 0]) / (Spot[ss + 1] - Spot[ss - 1])
            LHS      = -r * u[ss, 0] + (r - q) * Spot[ss] * DerS + kappa * theta * DerV
            U[ss, 0] = LHS * dt + u[ss, 0]
        # Update the temporary grid u(s,t) with the boundary conditions
        u = U
        # Interior points of the grid (non boundary).
        # As usual previous time step values are in the temporary grid u(s,t)
        for s in range(1, NS - 1):
            for v in range(1, NV - 1):
                DerS = (u[s + 1, v] - u[s - 1, v]) / (Spot[s + 1] - Spot[s - 1])      # Central difference for dU/dS
                DerV = (u[s, v + 1] - u[s, v - 1]) / (Vol[v + 1] - Vol[v - 1])        # Central difference for dU/dV
                DerSS = ((u[s + 1, v] - u[s, v]) / \
                        (Spot[s + 1] - Spot[s]) - (u[s, v] - u[s - 1, v]) / \
                        (Spot[s] - Spot[s - 1])) / (Spot[s + 1] - Spot[s])            # d2U/dS2
                DerVV = ((u[s, v + 1] - u[s, v]) / \
                        (Vol[v + 1] - Vol[v]) - (u[s, v] - u[s, v - 1]) / \
                        (Vol[v] - Vol[v - 1])) / (Vol[v + 1] - Vol[v])                # d2U/dV2
                DerSV = (u[s + 1, v + 1] - u[s - 1, v + 1] - u[s + 1, v - 1] + u[s - 1, v - 1]) / \
                        (Spot[s + 1] - Spot[s - 1]) / (Vol[v + 1] - Vol[v - 1])       # d2U/dSdV
                L = 0.5 * Vol[v] * Spot[s] * Spot[s] * DerSS + rho * sigma * Vol[v] * Spot[s] * DerSV \
                  + 0.5 * sigma * sigma * Vol[v] * DerVV - r * u[s, v] \
                  + (r - q) * Spot[s] * DerS + kappa * (theta - Vol[v]) * DerV
                # The PDE
                U[s, v] = L * dt + u[s, v]
        
    U = U.transpose()        
        
    f = scipy.interpolate.interp2d(Spot, Vol, U)
    end = time.time()
    print('program ends...')
    return (f(Si, vi), end - start)    
#_______________________________________________________________________________________________________________________________
#
if __name__ == "__main__":

    # Heston Parameters
    p_heston = {'kappa':1.50000
               ,'theta':0.04000
               ,'sigma':0.30000
               ,'v0':0.05412
               ,'rho':-0.90000
               ,'lambda':0.00000}
    # Option features
    p_option = {'Strike':95
               ,'r':0.02
               ,'q':0.05
               }
    # Grid parameters
    p_grid   = {'NS':30
               ,'NV':30
               ,'NT':1500
               ,'Smin':0
               ,'Smax':2 * p_option['Strike']
               ,'Vmin':0
               ,'Vmax':0.5
               ,'Tmin':0
               ,'Tmax':0.15
               }    
    
    Si = 101.52000
    vi = 000.05412
    
    call_price_0 =  HestonExplicitPDE_1_0(Si, vi, p_heston, p_option, p_grid)
    #call_price_1 =  HestonExplicitPDE_1_1(Si, vi, p_heston, p_option, p_grid)
    call_price_2 =  HestonExplicitPDE_NU_1_1(Si, vi, p_heston, p_option, p_grid)
    
    print(call_price_0[0], call_price_0[1])
    #print(call_price_1[0], call_price_1[1])
    print(call_price_2[0], call_price_2[1])
