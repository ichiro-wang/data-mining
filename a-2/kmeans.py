import numpy as np


class KMeans:
    random = "random"
    pp = "kmeans++"

    def __init__(self, n_clusters: int, init: str = random, max_iter=300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None  # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iter = 0
        clustering = np.zeros(X.shape[0])  # which cluster each data point belongs to
        old_clustering = clustering

        while iter < self.max_iter:
            # find distance of each data point in X to each centroid
            distances = self.euclidean_distance(X, self.centroids)

            clustering = np.argmin(distances, axis=1)

            self.update_centroids(clustering, X)

            if np.array_equal(clustering, old_clustering):
                break

            old_clustering = clustering.copy()

            iter += 1

        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        k = self.n_clusters
        new_centroids = np.zeros((k, X.shape[1]))

        # update each centroid
        for i in range(k):
            cluster_i = X[clustering == i]
            # make sure there are actually points assigned to this centroid
            if len(cluster_i) > 0:
                new_centroids[i] = cluster_i.mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]

        self.centroids = new_centroids

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        k = self.n_clusters

        if self.init == KMeans.random:
            # select k data points randomly without replacement
            self.centroids = X[
                np.random.choice(range(X.shape[0]), replace=False, size=k)
            ]
        elif self.init == KMeans.pp:
            # choose first centroid randomly
            self.centroids = X[
                np.random.choice(range(X.shape[0]), replace=False, size=1)
            ]

            for i in range(1, k):
                # find distance of each data point in X to each centroid
                distances = self.euclidean_distance(X, self.centroids)
                # find distance to nearest centroid for each data point in X
                min_distances = np.min(distances, axis=1)
                min_squared = np.square(min_distances)

                # probability directly proportional to distance of nearest centroid
                pdf = min_squared / np.sum(min_squared)

                # choose next centroid based on this probability and add to our selected centroids
                next_centroid = X[np.random.choice(range(X.shape[0]), p=pdf)]
                self.centroids = np.vstack([self.centroids, next_centroid])

        else:
            raise ValueError(
                'Centroid initialization method should either be "random" or "k-means++"'
            )

    def euclidean_distance(self, X1: np.ndarray, X2: np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """

        """
        (a - b)^2 = a^2 - 2ab + b^2
        """
        X1_sum_squared = np.sum(np.square(X1), axis=1, keepdims=True)
        X2_sum_squared = np.sum(np.square(X2), axis=1, keepdims=True)

        mul = np.dot(X1, X2.T)

        dist_squared = X1_sum_squared - 2 * mul + X2_sum_squared.T

        return np.sqrt(dist_squared)

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        distances = self.euclidean_distance(X, self.centroids)

        n = X.shape[0]
        silhouette_scores = np.zeros(n)

        # iterate over each data point
        for i in range(n):
            # find distance to closest cluster
            closest_cluster = clustering[i]
            a = distances[i, closest_cluster]

            # find distance to 2nd closest cluster
            other_distances = distances[i].copy()
            other_distances[closest_cluster] = (
                np.inf
            )  # we want to ignore the closest cluster
            second_closest_cluster = np.argmin(other_distances)
            b = distances[i, second_closest_cluster]

            silhouette_scores[i] = (b - a) / max(a, b)

        return silhouette_scores.mean()
