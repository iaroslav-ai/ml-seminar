import numpy as np
from backend import show_tensor

A = np.random.rand(*(5,10))
B = np.random.rand(*(5,10))

# math operations on arrays
B = -B
print("-B",B)
B = np.sin(B)
print("sin(B)", B)
print("exp(B)", np.exp(B))

# operations with arrays of same shape
C = A + B
#show_tensor({'A':A, 'B':B, 'A+B': C})

A = (A + np.sin(B) * C) ** A

# incompatible shapes
D = np.random.rand(*(4, 9))
try:
    A = A + D
except BaseException as ex:
    print(ex)
    #show_tensor({'A':A, 'D':D})

# operations with constants
A = -np.random.rand(*(5,10))
#show_tensor({'A':A, 'A+3':A+3.0})

# reduction operations (sum, min, max)
A = np.eye(*(5,10))
print(np.sum(A))

A0 = np.sum(A, axis=0)
A1 = np.sum(A, axis=1)
show_tensor({'A':A, 'A0':A0, 'A1':A1})
