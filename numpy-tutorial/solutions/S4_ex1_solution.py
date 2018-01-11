import numpy as np
import pandas as ps

Xy = ps.read_csv('winequality-red.csv', sep=';').as_matrix()

"""
Select all columns except for the last one in array Xy and assign it to variable X.
Select a subset of first 100 rows in X starting from the row #10.
Calculate mean for every column in X. Subtract this mean value from every column.
Calculate standard deviation of modified X. Divide values in every column by
corresponding st. dev. value in X.

Caclulate and print mean and deviation of the columns of modified X.
"""

X = Xy[:, :-1]
X = X[9:100]

m = np.mean(X, axis=0)
X = X - m

d = np.std(X, axis=0)
X = X / d

print(np.mean(X, axis=0))
print(np.std(X, axis=0))