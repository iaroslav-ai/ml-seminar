"""
For more details see
https://pandas.pydata.org/pandas-docs/stable/10min.html
"""

import pandas as pd

# Merging is useful when you have multiple data sources,
# correpsonding to the same observations.

frame_a = pd.DataFrame({
    'A': pd.Series(['a', 'b', 'c', 'd'], index=[0, 1, 2, 3]),
    'B': pd.Series(['e', 'f', 'g', 'h'], index=[0, 1, 2, 3])
})

frame_b = pd.DataFrame({
    'C': pd.Series([1, 2], index=[1, 2]),
})

print(frame_a)
print(frame_b)

# Concatenation of observations
R = pd.concat([frame_a, frame_b], axis=1, join='inner')
print(R)

R = pd.concat([frame_a, frame_b], axis=1, join='outer')
print(R)
