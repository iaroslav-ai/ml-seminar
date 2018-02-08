"""See more details at:
https://pandas.pydata.org/pandas-docs/stable/dsintro.html
"""

import pandas as pd
import numpy as np

# A dataset consisting of columns!
# Various ways to create:

# From series or arrays
df = pd.DataFrame(data={
    'A': ['!', '?', ';'],
    'B': pd.Series([3, 2, 1], index=[1, 2, 3]),
    'C': pd.Series(['a', 'b', 'c'], index=[1, 2, 3]),
})

print(df)

# From data file:
df = pd.read_csv('titanic.csv')

# Print top rows of DataFrame
print(df.head())

# Get the columns
print(df.columns)

# Print information about the column types
print(df.info())

# Get the information about DataFrame.
print(df.describe())

# many other methods here https://pandas.pydata.org/pandas-docs/stable/dsintro.html

# Convert to numpy array:
num = df.as_matrix()

# Convert to csv file:
# df.to_csv('deleteme.csv')

# Column operations
# Select the column
print(df['Pclass'])

# Select multiple columns
print(df[['Pclass', 'Age']])

# Assignment
df['Pclass'] = df['Age']
print(df['Pclass'])

# Delete the column
del df['Pclass']

# Select subset of DataFrame
print(df.loc[:3, ['Age', 'Gender']])
