import random
import math

class KMeans:
    """Very small KMeans implementation without numpy."""

    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def fit(self, X):
        self.centroids = random.sample(X, self.n_clusters)
        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                idx = self._closest_centroid(x)
                clusters[idx].append(x)
            new_centroids = []
            for pts in clusters:
                if pts:
                    new_centroids.append(self._mean_point(pts))
                else:
                    new_centroids.append(random.choice(X))
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids

    def predict(self, X):
        return [self._closest_centroid(x) for x in X]

    def _closest_centroid(self, x):
        distances = [self._euclid(x, c) for c in self.centroids]
        return distances.index(min(distances))

    def _euclid(self, a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _mean_point(self, pts):
        n = len(pts[0])
        mean = [0.0] * n
        for p in pts:
            for i in range(n):
                mean[i] += p[i]
        return [v / len(pts) for v in mean]
