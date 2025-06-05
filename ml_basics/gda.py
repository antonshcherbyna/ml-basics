import math

class GaussianDiscriminant:
    """Gaussian Discriminant Analysis without numpy."""

    def __init__(self):
        self.label_ids = []
        self.means = []
        self.covariances = []
        self.priors = []

    def fit(self, X, y):
        self.label_ids = sorted(set(y))
        n_features = len(X[0]) if X else 0
        self.means = [[0.0] * n_features for _ in self.label_ids]
        counts = [0 for _ in self.label_ids]
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
        # compute covariances
        self.covariances = [[[0.0]*n_features for _ in range(n_features)]
                             for _ in self.label_ids]
        for xi, yi in zip(X, y):
            lid = self.label_ids.index(yi)
            diff = [xi[j] - self.means[lid][j] for j in range(n_features)]
            for i in range(n_features):
                for j in range(n_features):
                    self.covariances[lid][i][j] += diff[i]*diff[j]
        for lid in range(len(self.label_ids)):
            if counts[lid] == 0:
                continue
            for i in range(n_features):
                for j in range(n_features):
                    self.covariances[lid][i][j] /= counts[lid]
        total = float(len(X)) if X else 1.0
        self.priors = [c/total for c in counts]

    # basic linear algebra helpers
    def _matmul_vec(self, mat, vec):
        return [sum(mat[i][j]*vec[j] for j in range(len(vec))) for i in range(len(mat))]

    def _dot(self, a, b):
        return sum(x*y for x, y in zip(a, b))

    def _transpose(self, mat):
        return [list(row) for row in zip(*mat)]

    def _inverse(self, mat):
        n = len(mat)
        aug = [row[:] + [float(i==j) for j in range(n)] for i, row in enumerate(mat)]
        for i in range(n):
            pivot = aug[i][i]
            if pivot == 0:
                for k in range(i+1, n):
                    if aug[k][i] != 0:
                        aug[i], aug[k] = aug[k], aug[i]
                        pivot = aug[i][i]
                        break
            if pivot == 0:
                raise ValueError("Singular matrix")
            for j in range(2*n):
                aug[i][j] /= pivot
            for k in range(n):
                if k == i:
                    continue
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor*aug[i][j]
        return [row[n:] for row in aug]

    def _determinant(self, mat):
        if len(mat) == 1:
            return mat[0][0]
        det = 0
        for c in range(len(mat)):
            sub = [row[:c] + row[c+1:] for row in mat[1:]]
            det += ((-1)**c) * mat[0][c] * self._determinant(sub)
        return det

    def _gauss(self, x, mean, cov):
        # add tiny value to diagonal to avoid singular matrices
        cov_reg = [row[:] for row in cov]
        for i in range(len(cov_reg)):
            cov_reg[i][i] += 1e-6
        inv = self._inverse(cov_reg)
        det = self._determinant(cov_reg)
        diff = [x[i]-mean[i] for i in range(len(x))]
        exponent = -0.5 * self._dot(diff, self._matmul_vec(inv, diff))
        coef = 1.0 / math.sqrt(((2*math.pi)**len(x)) * abs(det))
        return coef * math.exp(exponent)

    def _prob_class(self, x, cid):
        prob = self.priors[cid]*self._gauss(x, self.means[cid], self.covariances[cid])
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
