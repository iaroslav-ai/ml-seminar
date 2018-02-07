import pandas as ps
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.svm import SVC
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

feature = make_union(
    make_pipeline(ColumnSelector(0), OneHotEncoder()),
    make_pipeline(ColumnSelector(1), Imputer()),
    make_pipeline(ColumnSelector(2), Imputer()),
    make_pipeline(ColumnSelector(4), OneHotEncoder()),
)

#feature.fit(X)
#X = feature.transform(X)

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
}

gsearch = GridSearchCV(
    estimator=model,
    param_grid=[svr_grid],
    verbose=1,
    cv=3,
    n_jobs=-1
)

dummy = DummyClassifier(strategy='most_frequent')

# fit all the transformers and estimators in the pipeline
gsearch.fit(X_train, y_train)
#dummy.fit(X_train, y_train)

print(gsearch.best_params_)
print("Model score: ", gsearch.score(X_test, y_test))
#print("Dummy score: ", dummy.score(X_test, y_test))