"""
This script tests what accuracy can be gained on the credit approval dataset
"""

from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from noxer.preprocessing import ColumnSelector, OneHotEncoder

from searchgrid import set_grid, build_param_grid

import pandas as ps
import numpy as np

data = ps.read_csv('credit-screening.csv')
data = data.replace('?', 'NaN')
data = data.as_matrix()

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

category = lambda idx: make_pipeline(ColumnSelector(idx), OneHotEncoder())
number = lambda idx: make_pipeline(ColumnSelector(idx), Imputer())

features = make_union(
    category(0),
    number(1),
    number(2),
    category(3),
    category(4),
    category(5),
    category(6),
    number(7),
    category(8),
    category(9),
    number(10),
    category(11),
    category(12),
    number(13),
    number(14)
)

estimator = Pipeline([
    ('features', features),
    ('scaler', StandardScaler()),
    ('model', LinearSVC())
])

estimator = set_grid(
    estimator,
    model=[
        set_grid(
            DecisionTreeClassifier(),
            min_samples_split=[2 ** -i for i in range(1, 8)],
            max_depth=list(range(1, 21))
        ),

        set_grid(
            LinearSVC(),
            C=np.logspace(-6, 6, num=20)
        ),

        set_grid(
            GradientBoostingClassifier(),
            n_estimators=[2 ** i for i in range(1, 11)],
            learning_rate=np.logspace(-4, 0, num=10)
        ),
    ],
    scaler=[
        StandardScaler(),
        RobustScaler(),
    ]
)

param_grid = build_param_grid(estimator)

gsearch = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    verbose=1,
    cv=10,
    n_jobs=-1
)

model = gsearch
dummy = DummyClassifier(strategy='most_frequent')

model.fit(X_train, y_train)
dummy.fit(np.zeros((len(y_train), 1)), y_train)
print(model.best_params_)
print(model.score(X_test, y_test))
print(dummy.score(np.zeros((len(y_test), 1)), y_test))
