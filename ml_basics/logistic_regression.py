import math

class LogisticRegression:
    """Binary logistic regression using gradient descent."""

    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.coef_ = []
        self.intercept_ = 0.0

    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        self.coef_ = [0.0] * n_features
        self.intercept_ = 0.0
        for _ in range(self.epochs):
            for i in range(n_samples):
                z = self.intercept_
                for j in range(n_features):
                    z += self.coef_[j] * X[i][j]
                pred = self._sigmoid(z)
                error = pred - y[i]
                for j in range(n_features):
                    self.coef_[j] -= self.lr * error * X[i][j]
                self.intercept_ -= self.lr * error

    def predict_proba(self, X):
        probs = []
        for x in X:
            z = self.intercept_
            for j, val in enumerate(x):
                z += self.coef_[j] * val
            probs.append(self._sigmoid(z))
        return probs

    def predict(self, X):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]
