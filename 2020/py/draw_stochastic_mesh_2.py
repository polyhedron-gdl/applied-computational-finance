from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure(figsize=[10, 10])
#
# Tree properties
#
#
n = 4  # number of levels
b = 3  # branching parameter
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

x_old = int(b) * [int(x[0])]
y_old = int(b) * [int(y[0])]
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

plt.title('Fig. 2a - Bulding Trajectories',     fontsize=12)
plt.xlabel('Time Step',        fontsize=12)
plt.ylabel('Simulation Path',  fontsize=12)
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

plt.title('Fig. 2b - Bulding Interactions',     fontsize=12)
plt.xlabel('Time Step',        fontsize=12)
plt.ylabel('Simulation Path',  fontsize=12)

plt.grid(True)
plt.show()


