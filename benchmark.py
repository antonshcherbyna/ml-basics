import time

from ml_basics import LinearRegression, LogisticRegression, KMeans

try:
    from sklearn.linear_model import LinearRegression as SkLinearRegression
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _benchmark_linear_regression():
    X = [[i] for i in range(20)]
    y = [2 * i + 1 for i in range(20)]
    start = time.time()
    model = LinearRegression(lr=0.01, epochs=500)
    model.fit(X, y)
    ours = time.time() - start

    if SKLEARN_AVAILABLE:
        start = time.time()
        sk = SkLinearRegression()
        sk.fit(X, y)
        sk_time = time.time() - start
    else:
        sk_time = None
    return ours, sk_time


def _benchmark_logistic_regression():
    X = [[0], [1]] * 50
    y = [0, 1] * 50
    start = time.time()
    model = LogisticRegression(lr=0.1, epochs=500)
    model.fit(X, y)
    ours = time.time() - start

    if SKLEARN_AVAILABLE:
        start = time.time()
        sk = SkLogisticRegression(max_iter=500)
        sk.fit(X, y)
        sk_time = time.time() - start
    else:
        sk_time = None
    return ours, sk_time


def main():
    lin_ours, lin_sk = _benchmark_linear_regression()
    log_ours, log_sk = _benchmark_logistic_regression()

    print("LinearRegression - ours: %.4fs" % lin_ours)
    if lin_sk is not None:
        print("LinearRegression - sklearn: %.4fs" % lin_sk)
    else:
        print("sklearn not available for LinearRegression benchmark")

    print("LogisticRegression - ours: %.4fs" % log_ours)
    if log_sk is not None:
        print("LogisticRegression - sklearn: %.4fs" % log_sk)
    else:
        print("sklearn not available for LogisticRegression benchmark")


if __name__ == "__main__":
    main()
