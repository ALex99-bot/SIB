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
        self._var = np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        X = dataset.X
        cond = self._var > self.threshold  # array de boolianos
        idxs = [i for i in range(len(cond)) if cond[i]]
        X_trans = X[:, idxs]
        xnames = [dataset.xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline)


class KBestSelection:
    def __init__(self, function, k):
        self.func = function
        self.k = k

    def fit(self, dataset):
        self.f_score, self.p_value = self.func(dataset)

    def transform(self, dataset, inline=False):
        best_f = np.argsort(self.f_score)[-self.k:]
        X = dataset.X
        X_trans = X[:, best_f]
        xnames = [dataset.xnames[i] for i in best_f]
        if inline:
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else:
            from .dataset import Dataset
            return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline)


def f_classif(dataset):
    labels = np.unique(dataset.Y)
    subsets = [dataset.X[dataset.Y == i, :] for i in labels]
    return stats.f_oneway(*subsets)


def f_regression(dataset):
    correlation_coefficient = []
    for c in range(dataset.X.shape[1]):
        correlation_coefficient.append(stats.pearsonr(dataset.X[:, c], dataset.Y)[0])

    correlation_coefficient = np.array(correlation_coefficient)

    deg_of_freedom = dataset.Y.size - 2
    corr_coef_squared = correlation_coefficient ** 2
    f_statistic = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p_values = stats.f.sf(f_statistic, 1, deg_of_freedom)
    return f_statistic, p_values


if __name__ == "__main__":
    a = KBestSelection(f_classif, 10)
