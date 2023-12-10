# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array

class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
        the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
        and/or encapsulate the necessary mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Initialization: Randomly choose k data points as initial centroids
        rng = np.random.default_rng(self.random_state)
        self.cluster_centers_ = X[rng.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assignment step: Assign each data point to the closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # Update step: Update centroids based on the mean of assigned data points
            new_centers = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.allclose(self.cluster_centers_, new_centers):
                break

            self.cluster_centers_ = new_centers

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_

class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances (optional).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
        long as it does this, you may change the content of this method completely and/or encapsulate the necessary
        mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Initialization:
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        current_label = 0

        for i in range(X.shape[0]):
            if self.labels_[i] != 0:
                continue

            neighbors = np.linalg.norm(X - X[i], axis=1) < self.eps

            if np.sum(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                current_label += 1
                self.labels_[i] = current_label
                stack = [i]

                while stack:
                    current_point = stack.pop()
                    current_neighbors = np.linalg.norm(X - X[current_point], axis=1) < self.eps

                    for neighbor in np.nonzero(current_neighbors)[0]:
                        if self.labels_[neighbor] == 0:
                            self.labels_[neighbor] = current_label

                            if np.sum(np.linalg.norm(X - X[neighbor], axis=1) < self.eps) >= self.min_samples:
                                stack.append(neighbor)
                        elif self.labels_[neighbor] == -1:
                            self.labels_[neighbor] = current_label

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
