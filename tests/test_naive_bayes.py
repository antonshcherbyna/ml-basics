import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import NaiveBayes

def test_naive_bayes_predict():
    X = [[0], [1]]
    y = [0, 1]
    nb = NaiveBayes()
    nb.fit(X, y)
    preds = nb.predict(X)
    assert preds == y
