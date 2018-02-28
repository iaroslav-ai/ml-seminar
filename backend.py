import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FlattenNDim(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.reshape(X, (len(X), -1))
        return X


class ImputerNDim(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        Xc = np.zeros_like(X)
        I = ~np.isnan(X)
        Xc[I] = X[I]
        self.means = np.mean(Xc, axis=tuple(range(Xc.ndim-1)))
        return self

    def transform(self, X, y=None):
        Xc = np.zeros_like(X)
        Xc = Xc + self.means
        I = ~np.isnan(X)
        Xc[I] = X[I]
        return Xc


class StandardScalerNDim(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.means = np.mean(X, axis=tuple(range(X.ndim-1)))
        X = X - self.means
        self.std = np.std(X, axis=tuple(range(X.ndim - 1)))
        return self

    def transform(self, X, y=None):
        X = X - self.means
        X = X / self.std
        return X