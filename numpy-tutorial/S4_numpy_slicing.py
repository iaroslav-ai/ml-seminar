import numpy as np
from backend import show_tensor

A = np.zeros((5,10))

# access and set values of the array
print(A[0,1])

A[1] = 1.0
A[1,:] = 1.0
A[:,1] = -1

#show_tensor({'A':A, 'A[1]':A[1], 'A[:,0]':A[:,0]})

X = A[:3,1:]
#show_tensor({ 'A':A, 'slice': X})

A[2:4,2:7] = -3.0
#show_tensor({ 'A':A})

A = np.random.randn(*(9,10))

A[[2,3,5,8],:] = 3.0
A[[2,3,5,8],[1,4,3,6]] = -3.0

#show_tensor({ 'A':A })

mn = np.mean(A, axis=0) > 1.0
A[:,mn] = 0.0
show_tensor({ 'A':A, 'mn':mn })