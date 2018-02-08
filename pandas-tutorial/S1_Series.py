"""See more details at:
https://pandas.pydata.org/pandas-docs/stable/dsintro.html
"""

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

# check if some element is in Series index
print('a' in x)

# slicing also possible on index keys!
print(x['a':'c'])

# numpy functions also can work on Series
print(np.mean(x))

# vectorized operations are possible on Series
print(x + x ** 2)

# there is an automated label alignment
print(x[:-1] + x[1:])
