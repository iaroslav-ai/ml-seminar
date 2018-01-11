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

import matplotlib.pyplot as plt
y = Xy[:, -1]

plt.hist(y, label='Histogram of scores')
plt.xlabel('scores')
plt.ylabel('counts')
plt.show()

for feature_idx in range(Xy.shape[-1]):
    plt.title('X_%s->Y' % feature_idx)
    plt.scatter(Xy[:, feature_idx], y, label='Relation')
    plt.xlabel('X_%s value' % feature_idx)
    plt.ylabel('Output')
    plt.show()