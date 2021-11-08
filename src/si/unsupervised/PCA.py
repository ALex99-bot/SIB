import numpy as np
from si.util.scale import StandardScaler
from copy import copy


def EVD(X, n_components):
    # Centrar os dados
    X_mean = X - np.mean(X, axis=0)
    # calculating the covariance matrix of the mean-centered data.
    cov_mat = np.cov(X_mean, rowvar=False)  # Não sei se é F ou T
    # Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # Sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # Select the first n eigenvectors, n is desired dimension of the final reduced data
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    # Transformação dos dados
    X_reduced = np.dot(eigenvector_subset.transpose(), X_mean.transpose()).transpose()
    return X_reduced, eigen_values


def SVD(X, n_components):
    # Center X and get covariance matrix C
    n, p = X.shape
    X -= X.mean(axis=0)
    # SVD
    u, sigma, vt = np.linalg.svd(X, full_matrices=False)
    # Return principal components and eigenvalues to calculate the portion of sample variance explained
    return np.dot(X, vt.T)[:, 0:n_components], (sigma ** 2) / (n - 1)


class PCA:
    def __init__(self, n_components, function=SVD):
        self.n_components = n_components
        self.func = function

    def varianve_explained(self, X):
        x_ = X - X.mean(axis=0)
        u, sigma, vt = np.linalg.svd(x_, full_matrices=False)
        eig_val = sigma ** 2 / (x_.shape[0] - 1)
        ve = eig_val / eig_val.sum()
        return ve

    def fit(self, dataset):
        self.scaler = StandardScaler()
        self.scaler.fit(dataset)
        return self.scaler

    def transform(self, dataset, inline=False):
        centered = self.scaler.transform(dataset)
        PC, self.eigen = self.func(centered.X) # func passa a ser PC e EV
        self.variance_explained()
        if inline:
            dataset.X = PC
            return dataset, self.eigen
        else:
            from ..data import Dataset
            return Dataset(PC, copy(dataset.Y), copy(dataset.xnames), copy(dataset.yname)), self.eigen

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline)
