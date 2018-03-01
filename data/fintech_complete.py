import pandas as pd
import numpy as np

data = pd.read_csv('lumber.csv').as_matrix()
data = data[::-1]

data = data[-1000:]

Ix = np.array([0, 1, 2, 3, 4, 5])
Iy = np.array([6])

X, Y = [], []

while max(Iy) < len(data):
    x = data[Ix]
    price_yesterday = data[Ix[-2]][1]
    price_today = data[Ix[-1]][1]
    x = x[:, 1:]
    price = data[Iy][0, 1]

    price = 1 if price > price_today else -1

    X.append(x)
    Y.append(price)

    Ix = Ix + 1
    Iy = Iy + 1

X = np.array(X).astype('float')
Y = np.array(Y)

from backend import StandardScalerNDim, ImputerNDim, FlattenNDim
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
N = int(len(X)*0.7)
#X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train, X_test, y_train, y_test = X[:N], X[N:], Y[:N], Y[N:]

estimator = make_pipeline(
    ImputerNDim(),
    StandardScalerNDim(),
    FlattenNDim(),
    LinearSVC(dual=False, penalty='l2', C=2.0),
)
estimator.fit(X_train, y_train)
print("Model score:", estimator.score(X_test, y_test))
dummy = DummyClassifier('most_frequent')
dummy.fit(y_train[:, np.newaxis], y_train)
print("Dummy score:", dummy.score(y_test[:, np.newaxis], y_test))
# calculate another trivial dummy
y_triv = [1 if x[-1, 1] > x[-2, 1] else -1 for x in X_test]
from sklearn.metrics import accuracy_score
print("Dummy forecast:", accuracy_score(y_test, y_triv))

final = estimator.steps[-1][-1]
if isinstance(final, LinearSVC):
    weights = final.coef_
    print(np.reshape(weights, X_test[0].shape))

from sklearn.model_selection import permutation_test_score

score, permutation_scores, pvalue = permutation_test_score(
    estimator, X, Y,
    scoring="accuracy",
    n_permutations=100,
    n_jobs=-1,
    verbose=1
)

import matplotlib.pyplot as plt

print(pvalue)

plt.hist(permutation_scores)
plt.xlabel('Scores')
plt.ylabel('Counts')
plt.show()

