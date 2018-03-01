import numpy as np

Xy = np.load('sf.npy')

# feature #0: Gyroscope
# feature #1: Accelerometer
# feature #2: Output

import matplotlib.pyplot as plt

def plot_data(X, y_true=None, y_pred=None):
    plt.clf()
    I = np.arange(len(X))

    if y_true is not None:
        plt.plot(I, y_true, label='Outputs')

    if y_pred is not None:
        plt.plot(I, y_pred, label='Estimations')

    plt.plot(I, X[:, -1, 0], label='Gyro')
    plt.plot(I, X[:, -1, 1], label='Accel')
    plt.legend()
    plt.show()

n = 100
Ix = np.arange(0, n)
Iy = n + 1

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
from backend import FlattenNDim, StandardScalerNDim, ImputerNDim
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Input, Dense, LeakyReLU, Flatten, LSTM, Conv1D, MaxPool1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

def build_fn():
    model = Sequential()
    #model.add(LSTM(64, input_shape=X[0].shape))
    for i in range(3):
        model.add(Conv1D(32, kernel_size=3, padding='same', input_shape=X[0].shape))
        model.add(LeakyReLU())
        model.add(Conv1D(32, kernel_size=3, padding='same'))
        model.add(LeakyReLU())
        model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(),
        loss="mse",
        #metrics=['accuracy']
    )

    return model

model = KerasRegressor(build_fn, epochs=10)


N = int(len(X)*0.7)
#X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train, X_test, y_train, y_test = X[:N], X[N:], Y[:N], Y[N:]
estimator = make_pipeline(
    StandardScalerNDim(),
    make_pipeline(FlattenNDim(), LinearSVR()),
    #model
)
estimator.fit(X_train, y_train)
print("Model score:", estimator.score(X_test, y_test))

dummy = DummyRegressor()
dummy.fit(y_train[:, np.newaxis], y_train)
print("Dummy score:", dummy.score(y_test[:, np.newaxis], y_test))

y_pred = estimator.predict(X_test)
print("R2 score: %s" % r2_score(y_test, y_pred))
plot_data(X_test, y_test, y_pred)