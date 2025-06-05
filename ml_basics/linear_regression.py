class LinearRegression:
    """Simple linear regression using gradient descent without external deps."""

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.coef_ = []
        self.intercept_ = 0.0

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        self.coef_ = [0.0] * n_features
        self.intercept_ = 0.0
        for _ in range(self.epochs):
            for i in range(n_samples):
                pred = self.intercept_
                for j in range(n_features):
                    pred += self.coef_[j] * X[i][j]
                error = pred - y[i]
                for j in range(n_features):
                    self.coef_[j] -= self.lr * error * X[i][j]
                self.intercept_ -= self.lr * error

    def predict(self, X):
        preds = []
        for x in X:
            pred = self.intercept_
            for j, val in enumerate(x):
                pred += self.coef_[j] * val
            preds.append(pred)
        return preds
