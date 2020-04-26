from matplotlib import pyplot as plt
import numpy as np
from lib2to3.fixes import fix_next

fig = plt.figure(figsize=[5, 5])

n = 3
b = 3

x = []
y = []
for i in range(n):
    for j in range(b**i):
        y.append(int(b*j/(b**(i-1)))) 
        x.append(i)
    #print(x)
    #print(y)

j1 = 0
x_old = [0]
y_old = [0]


#questo va bene per lo stochastic mesh

for i in range(1, n):
    
    j1  = j1 + b**i
    max = j1 
    min = max + 1 - b**i
    
    x_new = x[min:max+1]
    y_new = y[min:max+1]
    
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
    
    #print(x_new)
    #print(y_new)
    
    print(x_plt)
    print(y_plt)
    plt.plot(x_plt, y_plt, 'bo-')

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