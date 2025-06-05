import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import PCA

def test_pca_transform():
    X = [[1,0], [0,1], [1,1]]
    pca = PCA()
    pca.fit(X, 1)
    transformed = pca.transform(X)
    assert len(transformed[0]) == 1
    assert len(transformed) == len(X)
