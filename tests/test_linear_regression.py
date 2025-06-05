import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import LinearRegression

def test_linear_regression_fit_predict():
    X = [[1], [2], [3]]
    y = [3, 5, 7]  # y = 2*x + 1
    model = LinearRegression(lr=0.01, epochs=1000)
    model.fit(X, y)
    preds = model.predict(X)
    for p, t in zip(preds, y):
        assert abs(p - t) < 1
