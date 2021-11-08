import numpy as np


class PCA:
    def __init__(self, n_components=2, ):
        pass

    def PCA(self, X, n_components):
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

        return X_reduced

    def transform(self, dataset):
        pass

