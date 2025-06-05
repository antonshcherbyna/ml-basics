import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import KMeans

def test_kmeans_clusters():
    data = [[0], [0.1], [10], [10.1]]
    km = KMeans(n_clusters=2, max_iter=10)
    km.fit(data)
    preds = km.predict(data)
    assert len(set(preds)) == 2
