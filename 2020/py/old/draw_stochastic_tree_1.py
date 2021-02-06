from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure(figsize=[10, 10])
#
# Tree properties
#
#
n = 2  # number of levels
b = 3  # branching parameter
# 
# calculating tree points coordinates
#
x       = (b**n)*[n]
y       = range(b**n)
y_last  = y
for i in range(n-1,-1,-1):
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
# building graph
#
imax  = 0
x_old = [x[0]]
y_old = [y[0]]
for i in range(1, n+1):
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
            plt.plot(np.array(x_plt), np.array(y_plt), 'bo-')
    x_old = x_new
    y_old = y_new

plt.grid(True)
plt.show()
