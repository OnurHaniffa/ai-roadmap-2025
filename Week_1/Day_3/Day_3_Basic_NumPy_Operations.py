#Basic NumPy Operations and learting the notations 

import numpy as np
import matplotlib.pyplot as plt
array = np.array([4,8,15,16,23,42])
print(array)


array2 = np.random.randint(1,101,size=(3,3))
print(array2.shape, array2.dtype, array2.ndim, array2.size)

array3 = np.array([10,20,30,40,50])
print(array3[[0,1,2]])
print(array3[[-1,-2]])
print(array[[0,1,3,4]])

array4 = np.random.randint(1,51,size=10)
array4= array4*2
array4= array4[array4 < 60]
array4 = np.where(array4%2 == 0, -1, array4)


array5=np.random.randint(0,100,size=(5,5))
print(array5.mean(), array5.max(), array5.min(), array5.sum(), array5.std(), array5.sum(axis=0), array5.sum(axis=1))

array6=np.ones((3,3))
array6 = array6 + [5,10,15]
print(array6)

array7 = np.random.randint(0,1000,size=1000)
plt.hist(array7, bins=30, edgecolor='black')
plt.title('Histogram of Random Integers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()