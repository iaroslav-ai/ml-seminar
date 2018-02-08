# typical import of pandas
import pandas as pd
import numpy as np

# Series ~ a column of a dataset
x = pd.Series([-1, 1, -2, 2])
print(x)

# indexing
print(x[0])
print(x[0:2])

# Series can act like a dictionary
x = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

print(x)
print(x['a'])
print(x['a':'c'])