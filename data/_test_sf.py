import numpy as np

Xy = np.load('sf.npy')

# feature #0: Gyroscope
# feature #1: Accelerometer
# feature #2: Output

X = Xy[:, :2]
y = Xy[:, 2]

import matplotlib.pyplot as plt

def plot_data(X, y_true=None, y_pred=None):
    I = np.arange(len(X))

    if y_true is not None:
        plt.plot(I, y_true, label='Outputs')

    if y_pred is not None:
        plt.plot(I, y_pred, label='Estimations')

    plt.plot(I, X[:, 0], label='Gyro')
    plt.plot(I, X[:, 1], label='Accel')
    plt.legend()
    plt.show()


Ix = np.arange(0, 30)
Iy = 31

X, Y = [], []
data = Xy

while Iy < len(data):
    x = data[Ix]
    x = x[:, :2]
    y = data[Iy][2]

    X.append(x)
    Y.append(y)

    Ix = Ix + 1
    Iy = Iy + 1

X = np.array(X)
Y = np.array(Y)

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from backend import FlattenNDim
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
N = int(len(X)*0.7)
#X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train, X_test, y_train, y_test = X[:N], X[N:], Y[:N], Y[N:]
estimator = make_pipeline(
    FlattenNDim(),
    StandardScaler(),
    LinearSVR(),
)
estimator.fit(X_train, y_train)
print("Model score:", estimator.score(X_test, y_test))

dummy = DummyRegressor()
dummy.fit(y_train[:, np.newaxis], y_train)
print("Dummy score:", dummy.score(y_test[:, np.newaxis], y_test))

y_pred = estimator.predict(X_test)
plot_data(X_test, y_test, y_pred)