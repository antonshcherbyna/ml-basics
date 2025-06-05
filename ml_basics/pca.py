import math

class PCA:
    """Very small PCA implementation using power iteration."""

    def __init__(self):
        self.components = []
        self.mean = []

    def _center(self, X):
        n_features = len(X[0])
        self.mean = [0.0]*n_features
        for x in X:
            for j in range(n_features):
                self.mean[j] += x[j]
        self.mean = [m/len(X) for m in self.mean]
        centered = [[x[j]-self.mean[j] for j in range(n_features)] for x in X]
        return centered

    def _covariance(self, X):
        n = len(X)
        d = len(X[0])
        cov = [[0.0]*d for _ in range(d)]
        for x in X:
            for i in range(d):
                for j in range(d):
                    cov[i][j] += x[i]*x[j]
        for i in range(d):
            for j in range(d):
                cov[i][j] /= n
        return cov

    def _matmul_vec(self, mat, vec):
        return [sum(mat[i][j]*vec[j] for j in range(len(vec))) for i in range(len(mat))]

    def _normalize(self, vec):
        norm = math.sqrt(sum(v*v for v in vec))
        if norm == 0:
            return vec
        return [v/norm for v in vec]

    def fit(self, X, n_components):
        centered = self._center(X)
        cov = self._covariance(centered)
        comps = []
        for _ in range(n_components):
            v = [1.0]*len(cov)
            v = self._normalize(v)
            for _ in range(20):  # power iterations
                v = self._matmul_vec(cov, v)
                v = self._normalize(v)
            comps.append(v)
            # deflate
            lam = sum(v[i]*self._matmul_vec(cov, v)[i] for i in range(len(v)))
            for i in range(len(cov)):
                for j in range(len(cov)):
                    cov[i][j] -= lam*v[i]*v[j]
        self.components = comps

    def transform(self, X):
        result = []
        for x in X:
            centered = [x[j]-self.mean[j] for j in range(len(x))]
            new = []
            for comp in self.components:
                new.append(sum(centered[j]*comp[j] for j in range(len(comp))))
            result.append(new)
        return result
