import pandas as pd
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
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

# example pipeline
model = make_pipeline(
    StandardScaler(),
    SVR()
)
# print the parameters of the model
print(model.get_params())

gsearch = GridSearchCV(
    estimator=model,
    param_grid={
        "svr__C": np.logspace(-6, 6, num=20),
    },
    verbose=1,
    cv=3,
    n_jobs=-1
)
# fit all the transformers and estimators in the pipeline
gsearch.fit(X_train, y_train)
print(gsearch.score(X_test, y_test))

for i in range(10):
    ip = [X_test[i]]
    y_true = y_test[i]
    print(gsearch.predict(ip), y_true)