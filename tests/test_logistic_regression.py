import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import LogisticRegression

def test_logistic_regression_fit_predict():
    X = [[0], [1]]
    y = [0, 1]
    model = LogisticRegression(lr=0.1, epochs=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds == y
