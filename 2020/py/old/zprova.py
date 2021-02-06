import random
import numpy as np

b = 10


test=random.sample(range(1, 10000), b*b)
print(test)
print('------------------------------------------------------------------')    

x = range(0, b*b)
for i in range(0, b):
    for j in range(0, b/2):
        y1 = np.asarray([b*i + j, b*i + j + b/2])
        y2 = np.delete(x, y1)
        
        s1 = np.sum([test[j] for j in y1])/2
        s2 = np.sum([test[j] for j in y2])/(b-2)
        
        print(y1 , s1, s2)
    print('------------------------------------------------------------------')    
