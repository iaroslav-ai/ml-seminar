import pandas as ps
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

data = ps.read_csv('data/credit-screening.csv', header=None)
data = data.as_matrix()

X, y = data[:, :-1], data[:, -1]

lb = LabelBinarizer()
lb.fit(X[:, 0])

X1 = lb.transform(X[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X1, y, random_state=0)

model = Pipeline([
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

# fit all the transformers and estimators in the pipeline
gsearch.fit(X_train, y_train)
print(gsearch.best_params_)
print(gsearch.score(X_test, y_test))