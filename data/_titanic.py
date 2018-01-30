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


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Wrapper around LabelBinarizer. Assumes that input X to fit and transform is a single
    column matrix of categorical values."""
    def fit(self, X, y=None):
        # create label encoder
        M = X[:, 0].astype('str')
        self.encoder = LabelBinarizer()
        self.encoder.fit(M)
        return self

    def transform(self, X, y=None):
        M = X[:, 0].astype('str')
        return self.encoder.transform(M)


data = ps.read_csv('titanic.csv')
data = data.replace('?', 'NaN')
data = data.as_matrix()

X = data[:, :-1]
y = data[:, -1].astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

category = lambda idx: make_pipeline(ColumnSelector(idx), OneHotEncoder())
number = lambda idx: make_pipeline(ColumnSelector(idx), Imputer())

features = make_union(
    number(1), # Pclass
    category(3), # Gender
    number(4), # Age
    number(5), # SibSp
    category(6), # Parch
    number(8), # Fare
    category(10), # Embarked
)

estimator = Pipeline([
    ('features', features),
    ('scaler', StandardScaler()),
    ('model', LinearSVC())
])

models = [
        set_grid(
            DecisionTreeClassifier(),
            min_samples_split=[2 ** -i for i in range(1, 8)],
            max_depth=list(range(1, 21))
        ),

        set_grid(
            LinearSVC(),
            C=np.logspace(-6, 6, num=20)
        ),
]

estimator = set_grid(
    estimator,
    model=[

        set_grid(
            GradientBoostingClassifier(),
            n_estimators=[2 ** i for i in range(1, 11)],
            learning_rate=np.logspace(-4, 0, num=10)
        ),


    ],
    scaler=[
        StandardScaler()
    ]
)

param_grid = build_param_grid(estimator)

gsearch = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    verbose=1,
    cv=3,
    n_jobs=-1
)

model = gsearch
dummy = DummyClassifier(strategy='most_frequent')

model.fit(X_train, y_train)
dummy.fit(np.zeros((len(y_train), 1)), y_train)
print(model.best_params_)
print(model.score(X_test, y_test))
print(dummy.score(np.zeros((len(y_test), 1)), y_test))
