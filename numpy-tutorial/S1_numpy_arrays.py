import numpy as np

# creating array
lst = [1,2,3,4,5.0]
A = np.array(lst)

print("1d array", A)
print("dimensions of array", A.ndim)
print("shape of array", A.shape)

# creating multidimensional array
arr = [
    [1.0, 2.0],
    [3.0, 5.0]
]
A = np.array(arr)
print('2d array',A)

# generating arrays of zeros and ones
shape = (3,2)
z = np.zeros(shape)
print("array of all zeros", z)

one = np.ones(shape)

o_like = np.ones_like(z)
print("array of ones like z", o_like)

# random arrays
shape = (3,2)

# sampling from [0, 1]
r = np.random.rand(*shape)
print("uniform random array", r)

# sampling from normal distribution with zero mean and unit dev
r = np.random.randn(*shape)
print("normal distr. samples", r)

# linspace
print(np.linspace(0.0, 2.0, 10))

# copying arrays
B = np.copy(r)

# saving and loading numpy arrays
# https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
r = np.random.randn(*(4,4))
np.save('data',r)
print("saved the data")

data = np.load('data.npy')
print("loaded array:", data)