import numpy as np
from functools import reduce

class NaiveBayes:
    def __init__(self):
        self.data = None
        self.labels = None
        self.feat_ids = None
        self.label_ids = None
        self._means = None
        self._variances = None
        self._prior_probs = None

    def fit(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.feat_ids = [i for i in range(self.data.shape[1])]
        self.label_ids = sorted(set(labels))
        self._means = None
        self._variances = None
        self._prior_probs = None
        self.means
        self.variances
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
    def variances(self):
        if self._variances is None:
            self._variances = [
                [self.data[self.labels == label_id, feat_id].var() for feat_id in self.feat_ids]
                for label_id in self.label_ids
            ]
        return np.array(self._variances)

    @property
    def prior_probs(self):
        if self._prior_probs is None:
            self._prior_probs = [
                self.data[self.labels == label_id].size / self.data.size
                for label_id in self.label_ids
            ]
        return np.array(self._prior_probs)

    def gaus(self, x, mean, variance):
        exp = np.exp(-(x - mean)**2 / (2 * variance**2))
        return 1 / np.sqrt(2 * np.pi * variance**2) * exp

    def cond_prob(self, x, means, variances):
        probs = [self.gaus(_x, mean, variance) for _x, mean, variance in zip(x, means, variances)]
        return reduce(lambda m, n: m * n, probs)

    def predict(self, x):
        x = np.array(x)
        prediction = []
        for _x in x:
            probs = [prior_prob * self.cond_prob(_x, mean, variance)
                     for prior_prob, mean, variance in zip(self.prior_probs, self.means, self.variances)]
            probs = np.array(probs) / sum(probs)
            prediction.append(probs.argmax())
        return np.array(prediction)

    def predict_probs(self, x):
        x = np.array(x)
        prediction = []
        for _x in x:
            probs = [prior_prob * self.cond_prob(_x, mean, variance)
                     for prior_prob, mean, variance in zip(self.prior_probs, self.means, self.variances)]
            probs = np.array(probs) / sum(probs)
            prediction.append(probs)
        return np.array(prediction)
