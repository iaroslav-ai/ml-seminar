import numpy as np
import pandas as ps

Xy = ps.read_csv('winequality-red.csv', sep=';').as_matrix()

"""
Select all columns except for the last one in array Xy and assign it to variable X.
Calculate mean for every column in X. Subtract this mean value from every column.
Calculate standard deviation of modified X. Divide values in every column by
corresponding st. dev. value in X.

Caclulate and print mean and deviation of the values of modified X.
"""
