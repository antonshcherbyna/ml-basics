import math

class NaiveBayes:
    """Simple Gaussian Naive Bayes classifier without numpy."""

    def __init__(self):
        self.label_ids = []
        self.means = []
        self.variances = []
        self.priors = []

    def fit(self, X, y):
        self.label_ids = sorted(set(y))
        n_features = len(X[0]) if X else 0
        self.means = [[0.0] * n_features for _ in self.label_ids]
        self.variances = [[0.0] * n_features for _ in self.label_ids]
        counts = [0 for _ in self.label_ids]

        # accumulate sums
        for xi, yi in zip(X, y):
            lid = self.label_ids.index(yi)
            counts[lid] += 1
            for j, val in enumerate(xi):
                self.means[lid][j] += val
        for lid in range(len(self.label_ids)):
            if counts[lid] == 0:
                continue
            for j in range(n_features):
                self.means[lid][j] /= counts[lid]
        # compute variances
        for xi, yi in zip(X, y):
            lid = self.label_ids.index(yi)
            for j, val in enumerate(xi):
                diff = val - self.means[lid][j]
                self.variances[lid][j] += diff * diff
        for lid in range(len(self.label_ids)):
            if counts[lid] == 0:
                continue
            for j in range(n_features):
                self.variances[lid][j] /= counts[lid]
        total = float(len(X)) if X else 1.0
        self.priors = [c / total for c in counts]

    def _gauss(self, x, mean, var):
        if var == 0:
            return 1.0 if x == mean else 0.0
        coef = 1.0 / math.sqrt(2.0 * math.pi * var)
        exp = math.exp(-((x - mean) ** 2) / (2 * var))
        return coef * exp

    def _prob_class(self, x, cid):
        prob = self.priors[cid]
        for j, val in enumerate(x):
            prob *= self._gauss(val, self.means[cid][j], self.variances[cid][j])
        return prob

    def predict(self, X):
        preds = []
        for x in X:
            probs = [self._prob_class(x, cid) for cid in range(len(self.label_ids))]
            if sum(probs) == 0:
                preds.append(self.label_ids[0])
            else:
                preds.append(self.label_ids[probs.index(max(probs))])
        return preds
