from .util import train_test_split
import numpy as np
import itertools


class CrossValidationScore:

    def __init__(self, model, dataset, score=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.score = score
        self.cv = kwargs.get("cv", 3)
        self.split = kwargs.get("split", 0.8)
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for _ in range(self.cv):
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if not self.score:
                train_scores.append(self.model.cost())
                test_scores.append(self.model.cost(test.X, test.y))
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis=0, arr=train.X.T)
                train_scores.append(self.score(train.y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis=0, arr=test.X.T)
                train_scores.append(self.score(train.y, y_train))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores

    def to_data_frame(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, "Need to run"
        return pd.DataFrame({"Train Scores": self.train_scores, "Test Score": self.test_scores})


class GridSearchCV:
    def __init__(self, model, dataset, parameters, **kwargs):
        self.model = model
        self.dataset = dataset
        hasparam = [hasattr(self.model, param) for param in parameters]
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError(f"Wrong parameters: {keys[index]}")
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        for conf in itertools.product(*values):
            for i in range(len(attrs)):
                setattr(self.model, attrs[i], conf[i])
            scores = CrossValidationScore(self.model, self.dataset, **self.kwargs).run()
            self.results.append((conf, scores))
        return self.results

    def to_data_frame(self):
        import pandas as pd
        assert self.results, 'The grid search needs to be ran first'
        data = dict()
        for i, k in enumerate(self.parameters.keys()):
            v = list()
            for r in self.results:
                v.append(r[0][i])
            data[k] = v
        for i in range(len(self.results[0][1][0])):
            train, test = list(), list()
            for res in self.results:
                train.append(res[1][0][i])
                test.append(res[1][1][i])
            data['Train' + str(i + 1)] = train
            data['Test' + str(i + 1)] = test
        return pd.DataFrame(data)

