import pandas as pd
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np

# reading the data as numpy array
Xy = pd.read_csv('data/plant.csv').as_matrix()

# splitting the data into inputs and outputs
X = Xy[:, :-1]
y = Xy[:, -1]

# random split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# example pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('est', SVR())
])

# print the parameters of the model
print(model.get_params())

knn_grid = {
    'est': [KNeighborsRegressor()],
    'est__n_neighbors': list(range(1, 200))
}
gbm_grid = {
    'est': [GradientBoostingRegressor()],
    'est__n_estimators': [8, 32, 128],
    'est__learning_rate': [0.01, 0.1, 1.0]
}
svr_grid = {
    'est': [SVR()],
    'est__C': [0.1, 1.0, 10.0],
    'est__gamma': [0.1, 1.0],
}

gsearch = GridSearchCV(
    estimator=model,
    param_grid=[knn_grid, svr_grid, gbm_grid],
    verbose=1,
    cv=3,
    n_jobs=-1
)

# fit all the transformers and estimators in the pipeline
gsearch.fit(X_train, y_train)
print(gsearch.best_params_)
print(gsearch.score(X_test, y_test))

"""
for i in range(10):
    ip = [X_test[i]]
    y_true = y_test[i]
    print(gsearch.predict(ip), y_true)
"""