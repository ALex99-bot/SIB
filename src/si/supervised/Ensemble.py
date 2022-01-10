from .model import Model
import numpy as np
import pandas as pd


def majoraty(values):
    return max(set(values), key=values.count)


def average(values):
    return sum(values) / len(values)


class Ensemble(Model):
    def __init__(self, models, fvote, score):
        super.__init__()
        self.models = models
        self.fvote = fvote
        self.score = score

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        assert self.is_fitted, "Model must be fit before predicting."
        preds = [model.predict(x) for model in self.models]
        vote = self.fvote(preds)
        return vote

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return self.score(y, y_pred)


class ConfusionMatrix:
    def __init__(self, true_y, pred_y):
        self.true_y = np.array(true_y)
        self.pred_y = np.array(pred_y)
        self.conf = None

    def calc(self):
        self.conf = pd.crosstab(self.true_y, self.pred_y, rownames=['Actual'], colnames=['Predicted'], margins=True)

    def to_data_frame(self):
        return self.conf

    def __call__(self):
        self.calc()
