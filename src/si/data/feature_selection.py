import numpy as np
from scipy import stats
from copy import copy
import warnings


class VarianceThreshold:

    def __init__(self, threshold=0):
        if threshold < 0:
            warnings.warn("The threshold must be a non-negative value.")
        self.threshold = threshold

    def fit(self, dataset):
        X = dataset.X
        self.var = np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        X = dataset.X
        cond = self._var > self.threshold # array de boolianos
        idxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:, idxs]
        xnames = [dataset.xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset.yname))
