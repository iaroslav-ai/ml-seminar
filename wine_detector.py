import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
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
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

# example pipeline
model = make_pipeline(
    make_pipeline(
        StandardScaler(with_std=False),
        RobustScaler()
    ),
    SVR()
)

# print the parameters of the model
print(model.get_params())

# set the parameters of the pipeline
model.set_params(
    pipeline__standardscaler__with_std=False,
    svr__C=0.1
)

# fit all the transformers and estimators in the pipeline
model.fit(X_train, y_train)

# save the model
import pickle as pc
pc.dump(model, open('model.bin', 'wb'))

# load the model (possibly in a different module)
model = pc.load(open('model.bin', 'rb'))

# make predictions with the model and estimate the quality of the model
print(model.predict(X))
print(model.score(X_test, y_test))