from ..util.util import sigmoid, add_intersect
import numpy as np


class LogisticRegression:
    """Regressão logística sem regularização"""
    def __init__(self, epochs=1000, lr=0.1):
        super(LogisticRegression, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.theta = None

    def fit(self, dataset):
        X, y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.y = y
        # Closed form or GD
        self.train(X, y)
        self.is_fitted = True

    def train(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            grad = np.dot((h-y), X.T)/y.size
            self.theta -= self.lr*grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'Model must be fit before predicting'
        x = np.hstack(([1], X))
        p = sigmoid(np.dot(self.theta, x))

        if p <= 0.5:
            res = 0
        else:
            res = 1
        return res

    def cost(self,  X=None, y=None, theta=None):
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta
        m = X.shape[0]
        n = X.shape[1]
        h = sigmoid(np.dot(self.X, self.theta))
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h))
        res = np.sum(cost) / m
        return res


class LogisticRegressionReg:
    """Regressão logística com regularização"""
    def __init__(self, epochs=1000, lr=0.1):
        super(LogisticRegression, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.theta = None

    def fit(self, dataset):
        X, y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.y = y
        # Closed form or GD
        self.train(X, y)
        self.is_fitted = True

    def train(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            grad = np.dot((h-y), X.T)/y.size
            self.theta -= self.lr*grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'Model must be fit before predicting'
        x = np.hstack(([1], X))
        p = sigmoid(np.dot(self.theta, x))

        if p <= 0.5:
            res = 0
        else:
            res = 1
        return res

    def cost(self,  X=None, y=None, theta=None):
        X = add_intersect(X) if X is not None else self.X
        y = y if y is not None else self.y
        theta = theta if theta is not None else self.theta
        m = X.shape[0]
        n = X.shape[1]
        h = sigmoid(np.dot(self.X, self.theta))
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h))
        res = np.sum(cost) / m
        return res