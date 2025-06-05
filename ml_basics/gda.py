import numpy as np

class GaussianDiscriminant:
    def __init__(self):
        self.data = None
        self.labels = None
        self.feat_ids = None
        self.label_ids = None
        self._means = None
        self._covariances = None
        self._prior_probs = None

    def fit(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.feat_ids = [i for i in range(self.data.shape[1])]
        self.label_ids = sorted(set(labels))
        self._means = None
        self._covariances = None
        self._prior_probs = None
        self.means
        self.covariances
        self.prior_probs

    @property
    def means(self):
        if self._means is None:
            self._means = [
                [self.data[self.labels == label_id, feat_id].mean() for feat_id in self.feat_ids]
                for label_id in self.label_ids
            ]
        return np.array(self._means)

    @property
    def covariances(self):
        if self._covariances is None:
            self._covariances = [
                np.cov(self.data[self.labels == label_id], rowvar=False)
                for label_id in self.label_ids
            ]
        return np.array(self._covariances)

    @property
    def prior_probs(self):
        if self._prior_probs is None:
            self._prior_probs = [
                self.data[self.labels == label_id].size / self.data.size
                for label_id in self.label_ids
            ]
        return np.array(self._prior_probs)

    def gauss(self, x, mean, covariance):
        exp = np.exp(-0.5 * np.matmul(x - mean, np.matmul(np.linalg.pinv(covariance), x - mean)))
        return 1 / ((2 * np.pi) ** (len(x) / 2) * np.linalg.det(covariance) ** 0.5) * exp

    def cond_probs(self, x):
        return np.array([self.gauss(x, mean, cov) for mean, cov in zip(self.means, self.covariances)])

    def predict(self, x):
        x = np.array(x)
        prediction = []
        for _x in x:
            probs = self.cond_probs(_x) * self.prior_probs
            probs = probs / probs.sum()
            prediction.append(probs.argmax())
        return np.array(prediction)

    def predict_probs(self, x):
        x = np.array(x)
        prediction = []
        for _x in x:
            probs = self.cond_probs(_x) * self.prior_probs
            probs = probs / probs.sum()
            prediction.append(probs)
        return np.array(prediction)
