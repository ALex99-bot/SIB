import numpy as np
from si.util import util, metrics
from model import Model


class KNN(Model):
    def __init__(self,k , num_neighbors):
        super(KNN).__init()
        self.k = k
        self.num_neighbors = num_neighbors

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        # distância euclidiana
        distances = l2_distance(x, dataset.X)
        sorted_index = np.argsort(distances)
        return sorted_index[:self.num_neighbors]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before prediction.'
        neighbors = self.get_neighbors(x)
        values = self.dataset.y[neighbors].tolist()
        prediction =max(set(values), key=values.count)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        return accuracy_score(self.dataset.y, y_pred)