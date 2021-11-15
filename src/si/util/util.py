import itertools

import numpy as np
import pandas as pd

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']


def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.hasLabel():
        fullds = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y), 1)))
        columns = dataset.xnames[:]+[dataset.yname]
    else:
        fullds = dataset.X
        columns = dataset.xnames[:]
    _means = np.mean(fullds, axis=0)
    _vars = np.var(fullds, axis=0)
    _mins = np.min(fullds, axis=0)
    _maxs = np.max(fullds, axis=0)
    stats = {}
    for i in range(fullds.shape[1]):
        stat = {'mean': _means[i],
                'var': _vars[i],
                'min': _mins[i],
                'max': _maxs[i]
                }
        stats[columns[i]] = stat
    if format == "df":
        import pandas as pd
        df = pd.DataFrame(stats)
        return df
    else:
        return stats


def l1_distance(x, y):
    dist = (np.absolute(x-y)).sum(axis=1)
    return dist


def l2_distance(x, y):
    dist = ((x - y)**2).sum(axis=1)
    return dist


def train_test_split(dataset, split=0.8):
    n = dataset.X.shape[0]
    m = int(split*n)
    arr = np.arange(n)
    np.random.shuffle(arr)
    from ..data import Dataset
    train = Dataset(dataset.X[arr[:n]], dataset.y[arr[:n]], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[arr[m:]], dataset.y[arr[m:]], dataset._xnames, dataset._yname)
    return train, test


def sigmoid(z):
    return 1/(1+np.exp(-z))