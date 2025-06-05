import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import GaussianDiscriminant

def test_gda_predict():
    X = [[0,0], [0,1], [1,1], [1,0]]
    y = [0,0,1,1]
    gda = GaussianDiscriminant()
    gda.fit(X, y)
    preds = gda.predict(X)
    assert preds == y
