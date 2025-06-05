import numpy as np

class PCA:
    def __init__(self):
        self.reduction_mat = None

    def fit(self, data, num_components):
        data = np.array(data)
        data = data - data.mean(axis=0)
        eig_vecs = np.linalg.svd(np.cov(data, rowvar=False))[0]
        self.reduction_mat = eig_vecs[:, :num_components]

    def transform(self, x):
        transformed_x = [np.matmul(self.reduction_mat.T, _x) for _x in x]
        return np.array(transformed_x)

class KMeans:
    def __init__(self):
        self.labels = None

    def clusterize(self, x, num_clusters):
        x = np.array(x)
        num_samples = len(x)
        init_ids = np.random.randint(0, num_samples, num_clusters)
        means = np.array([x[init_id] for init_id in init_ids])
        labels = np.zeros(num_samples, dtype=np.int32)
        while True:
            for i, _x in enumerate(x):
                dists = [np.linalg.norm(_x - mean)**2 for mean in means]
                labels[i] = np.argmin(dists)
            new_means = [x[labels == i].mean(axis=0) for i in range(num_clusters)]
            new_means = np.array(new_means)
            if np.all([np.allclose(new_means[i], means[i]) for i in range(num_clusters)]):
                self.labels = labels
                break
            means = new_means
