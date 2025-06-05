import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import LinearSVM

def test_svm_fit_predict():
    X = [[0], [1]]
    y = [-1, 1]
    svm = LinearSVM(num_iter=500)
    svm.fit(X, y)
    preds = svm.predict(X)
    assert preds == y
