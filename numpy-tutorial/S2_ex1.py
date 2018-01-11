import numpy as np
import pandas as ps

Xy = ps.read_csv('winequality-red.csv', sep=';').as_matrix()

"""
Compute minimum and maximum values for every column in Xy and print them.
Compute the mean and standard deviation of the last column, and
print it in the console.
"""