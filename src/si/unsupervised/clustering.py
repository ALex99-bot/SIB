import numpy as np


class KMeans:
    def __init__(self, k: int, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, dataset):
        x = dataset.X
        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)

    def init_centroids(self):
        self.centroids = np.array([np.random.uniform(low=self._min, high=self._max) for i in range(self.k)])

    def init_centroids_random_points(self, X):
        rng = np.random.default_rng()
        self.centroids = rng.choice(X, self.k)
        return self.centroids

    def distance(self, x, centroids):
        return np.sqrt(((centroids-x)**2).sum(axis=1))

    def get_closest_centroids(self, x):
        dist = self.distance(x, self.centroids)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        X = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(X.shape[0])
        self.init_centroids_random_points(X)
        while changed and count < self.max_iterations:
            # array of indexes of nearest centroid
            idxs = np.apply_along_axis(self.get_closest_centroids, axis=1, arr=X)
            cent = [np.mean(X[idxs == i], axis=0) for i in range(self.k)]
            self.centroids = np.array(cent)
            changed = np.all(old_idxs == idxs)
            old_idxs = idxs
            count += 1
        return self.centroids, idxs
