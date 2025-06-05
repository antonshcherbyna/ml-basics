import random
import math

class LinearSVM:
    """Linear SVM using Pegasos optimizer."""

    def __init__(self, lam=0.0001, num_iter=1000):
        self.lam = lam
        self.num_iter = num_iter
        self.weights = []  # includes bias as last element

    def fit(self, X, y):
        n_features = len(X[0]) if X else 0
        self.weights = [0.0]*(n_features+1)
        for t in range(1, self.num_iter):
            i = random.randrange(len(X))
            x = X[i] + [1.0]
            eta = 1.0/(self.lam*t)
            margin = sum(w*xj for w,xj in zip(self.weights,x))
            if y[i]*margin < 1.0:
                self.weights = [(1-1/t)*w + eta*y[i]*xj for w,xj in zip(self.weights,x)]
            else:
                self.weights = [(1-1/t)*w for w in self.weights]
            norm = math.sqrt(sum(w*w for w in self.weights[:-1]))
            if norm > 0:
                scale = min(1.0, 1.0/math.sqrt(self.lam)/norm)
                self.weights = [w*scale for w in self.weights]

    def predict(self, X):
        preds = []
        for x in X:
            val = sum(w*xj for w,xj in zip(self.weights,x+[1.0]))
            preds.append(1 if val >= 0 else -1)
        return preds
