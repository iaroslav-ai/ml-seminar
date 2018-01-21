import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# reading the data as numpy array
Xy = pd.read_csv('data/winequality-red.csv', ';').as_matrix()

# splitting the data into inputs and outputs
X = Xy[:, :-1]
y = Xy[:, -1]

# random split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

#model = SVR(C=0.6, gamma=0.01)
# example pipeline
model = make_pipeline(
    make_pipeline(
        StandardScaler(with_std=False),
        RobustScaler()
    ),
    SVR()
)

print(model.get_params())

model.set_params(
    pipeline__standardscaler__with_std=False,
    svr__C=0.1
)

model.fit(X_train, y_train)
print(model.predict(X))
print(model.score(X_test, y_test))

import pickle as pc
pc.dump(model, open('model.bin', 'wb'))

model = pc.load(open('model.bin', 'rb'))
print(model.score(X_test, y_test))


"""
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)

sc2 = RobustScaler()
sc2.fit(X_train)
X_train = sc2.transform(X_train)

from sklearn.metrics import r2_score

# Regressors: score range in [-Inf, 1.0]
print(model.score(X_train, y_train))

yp = model.predict(X_train)
print(r2_score(y_train, yp))

print(model.score(sc.transform(X_val), y_val))
print(model.score(sc.transform(X_test), y_test))
"""

