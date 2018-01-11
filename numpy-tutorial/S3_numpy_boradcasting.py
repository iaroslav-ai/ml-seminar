import numpy as np
from backend import show_tensor

# boradcasting
A = np.zeros((5,10))
B = np.random.rand(*(1,10))
C = np.random.rand(*(5,1))

show_tensor({'A':A, 'B':B, 'C':C, 'A+B': A+B, 'A+C': A+C})

# higher dimensions
A = np.zeros((3,3,3,3))
B = np.ones(3)
C = np.random.rand(*(1,3,3,1))

print("A+B", A+B)
print("A+C", A+C)