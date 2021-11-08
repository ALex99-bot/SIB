import numpy as np
from si.util.scale import StandardScaler
from copy import copy


def EVD(X, n_components):
    # calculating the covariance matrix of the mean-centered data.
    cov_mat = np.cov(X, rowvar=False)  # Não sei se é F ou T
    # Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # Sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # Select the first n eigenvectors, n is desired dimension of the final reduced data
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    # Transformação dos dados
    X_reduced = np.dot(eigenvector_subset.transpose(), X.transpose()).transpose()
    return X_reduced, eigen_values


def SVD(X, n_components):
    # SVD
    n, p = X.shape
    u, s_value, vt = np.linalg.svd(X, full_matrices=False)
    # Return principal components and eigenvalues to calculate the portion of sample variance explained
    return np.dot(X, vt.T)[:, 0:n_components], s_value


class PCA:
    def __init__(self, n_components, function=SVD):
        self.n_components = n_components
        self.func = function

    def variance_explained(self):
        ve = self.eigen / self.eigen.sum()
        return ve

    def fit(self, dataset):
        self.scaler = StandardScaler()
        self.scaler.fit(dataset)
        return self.scaler

    def transform(self, dataset):
        centered = self.scaler.transform(dataset)
        PC, self.eigen = self.func(centered.X, self.n_components) # func passa a ser PC e EV
        self.variance_explained()
        return PC

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
