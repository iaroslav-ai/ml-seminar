import pandas as pd
from sklearn.svm import SVR

# reading the data as numpy array
Xy = pd.read_csv('data/winequality-red.csv', ';').as_matrix()

# splitting the data into inputs and outputs
X = Xy[:, :-1]
y = Xy[:, -1]

X_train, X_val, X_test = X[:1000], X[1000:1200], X[1200:]
y_train, y_val, y_test = y[:1000], y[1000:1200], y[1200:]

model = SVR(C=0.7)
model.fit(X_train, y_train)

#print(model.predict(X))

# Regressors: score range in [-Inf, 1.0]
#print(model.score(X_train, y_train))
print(model.score(X_val, y_val))
print(model.score(X_test, y_test))