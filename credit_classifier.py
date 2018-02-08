import pandas as ps
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, make_union


# selects a single column from the dataset
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, index=0):
        self.index = index

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, [self.index]]


# one hot encodes
class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        lb = LabelBinarizer()
        v = X[:, 0]
        lb.fit(v)
        self.lb = lb
        return self

    def transform(self, X, y=None):
        v = X[:, 0]
        return self.lb.transform(v)


data = ps.read_csv('data/credit-screening.csv', header=None)
data = data.replace('?', 'NaN')
data = data.as_matrix()

X, y = data[:, :-1], data[:, -1]

def cat(idx):
    return make_pipeline(ColumnSelector(idx), OneHotEncoder())

def num(idx):
    return make_pipeline(ColumnSelector(idx), Imputer())

feature = make_union(
    cat(0),
    num(1),
    num(2),
    cat(3),
    cat(4),
    cat(5),
    cat(6),
    num(7),
    cat(8),
    cat(9),
    num(10),
    cat(11),
    cat(12),
    num(13),
    num(14),
)

#feature.fit(X)
#Xt = feature.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = Pipeline([
    ('features', feature),
    ('scaler', StandardScaler()),
    ('est', SVC())
])

svr_grid = {
    'est': [SVC()],
    'est__C': [0.1, 1.0, 10.0],
    'est__gamma': [0.1, 1.0],
    #'features__pipeline-1': [None, cat(0)],
}

knn_grid = {
    'est': [KNeighborsClassifier()],
    'est__n_neighbors': [5, 10, 15]
}

params = model.get_params()
print(params)

gsearch = GridSearchCV(
    estimator=model,
    param_grid=[svr_grid],
    verbose=0,
    cv=3,
    n_jobs=8
)

dummy = DummyClassifier(strategy='most_frequent')

# fit all the transformers and estimators in the pipeline
gsearch.fit(X_train, y_train)
dummy.fit(np.zeros((len(y_train), 1)), y_train)

print(gsearch.best_params_)
print("Model score: ", gsearch.score(X_test, y_test))
#print(gsearch.cv_results_)
print("Dummy score: ", dummy.score(np.zeros((len(y_test), 1)), y_test))