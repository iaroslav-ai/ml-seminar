import numpy as np
import pandas as ps

Xy = ps.read_csv('winequality-red.csv', sep=';')
print(list(enumerate(Xy.columns)))
Xy = Xy.as_matrix()

"""
Produce the histogram of values of last (output) column.
Examine the correlation of every separate feature with target value
by plotting the graph where the x axis is some feature, and y axis is
the target value. Visualize in a for loop all such relations in a plot.
"""