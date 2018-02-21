import pandas as pd
import numpy as np

data = pd.read_csv('data/lumber.csv').as_matrix()
data = data[::-1]

data = data[-1000:]

Ix = np.array([0, 1, 2, 3, 4, 5])
Iy = np.array([6])

X, Y = [], []

while max(Iy) < len(data):
    x = data[Ix]
    price_prev = data[Ix[-1]][1]
    x = x[:, 1:]
    price = data[Iy][0, 1]

    price = 1 if price > price_prev else -1

    X.append(x)
    Y.append(price)

    Ix = Ix + 1
    Iy = Iy + 1

X = np.reshape(X, (len(X), -1))
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
N = int(len(X)*0.7)
#X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train, X_test, y_train, y_test = X[:N], X[N:], Y[:N], Y[N:]
estimator = make_pipeline(
    Imputer(),
    StandardScaler(),
    GradientBoostingClassifier(n_estimators=2000),
)
estimator.fit(X_train, y_train)
print("Model score:", estimator.score(X_test, y_test))
dummy = DummyClassifier('most_frequent')
dummy.fit(X_train, y_train)
print("Dummy score:", dummy.score(X_test, y_test))