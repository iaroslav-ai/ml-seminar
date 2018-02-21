"""
A test is made for how well the lumber prices can be forecasted
"""

import pandas as ps
import numpy as np

from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier

class FlattenShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.reshape(X, (len(X), -1))
        return X


class SpectrumTransform(BaseEstimator, TransformerMixin):
    def __init__(self, max_feats):
        self.max_feats = max_feats

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.fft.fft(X, axis=1)
        X = X[:, :self.max_feats, :]
        X = abs(X)
        return X

data = ps.read_csv("lumber.csv")
data = data.iloc[-2000:]

lag = 6
Ix = np.arange(lag)
Iy = np.array([np.max(Ix)+1])

X, Y = [], []

while np.max(Iy) < len(data):
    x = data.iloc[Ix]
    del x['Date']
    last_day = data.iloc[Ix[-1]]['Open']
    y = data.iloc[Iy]['Open']

    y = float(last_day) < float(y)

    X.append(x.as_matrix())
    Y.append(y)

    Ix += 1
    Iy += 1

X = np.array(X)
Y = np.array(Y)

estimator = make_pipeline(
    #SpectrumTransform(-1),
    FlattenShape(),
    Imputer(),
    StandardScaler(),
    LinearSVC(max_iter=100000),
    #GradientBoostingClassifier(),
    #SVC(kernel='linear')
)

N = len(X)
train = int(N*0.7)
X_train, X_test, y_train, y_test = X[:train], X[train:], Y[:train], Y[train:]

val = int(train*0.7)
I_train = np.arange(0, val)
I_val = np.arange(val, train)

model = GridSearchCV(
    estimator=estimator,
    param_grid={
        #"linearsvc__C":  [1.0] #np.logspace(-3, 3, 20) #
    },
    n_jobs=-1,
    verbose=1,
    cv=[(I_train, I_val)]
)


#X_train, X_test, y_train, y_test = train_test_split(X, Y)
model.fit(X_train, y_train)

dummy = DummyClassifier("most_frequent")
dummy.fit(np.zeros((len(y_train), 1)), y_train)

print("Model score: ", model.score(X_test, y_test))
#print(gsearch.cv_results_)
print("Dummy score: ", dummy.score(np.zeros((len(y_test), 1)), y_test))
