from matplotlib import pyplot as plt
import numpy as np
from lib2to3.fixes import fix_next

fig = plt.figure(figsize=[5, 10])

n = 3
b = 3

x = [0]
y = [(b**(n-1)-1)/2]
for i in range(1,n):
    for j in range(b**i):
        lower = (b**(n-1)-1)/2 - b*(i+2)
        lower = max(lower, 0)
        delta = b**(n-i-1)
        #print(i, lower , lower + j*delta)
        y.append(lower + j*delta)
        x.append(i)
print(x)
print(y)

j1    = 0
x_old = [0]
y_old = [(b**(n-1)-1)/2]

for i in range(1, n):
    
    j1  = j1 + b**i
    imax = j1 
    imin = imax + 1 - b**i
    
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

'''
for i in range(2):
    x = [1, 0, 1]
    for j in range(i):
        x.append(0)
        x.append(1)
    x = np.array(x) + i
    y = np.arange(-(i+1), i+2)[::-1]
    print(x,y)
    plt.plot(x, y, 'bo-')

plt.show()
'''


'''
y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
n = [58, 651, 393, 203, 123]

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))
    
plt.show()   
''' 