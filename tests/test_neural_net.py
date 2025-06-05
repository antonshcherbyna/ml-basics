import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_basics import NeuralClassifier

def test_neural_net_predict():
    X = [[0], [1]]
    y = [0, 1]
    net = NeuralClassifier(num_inputs=1, num_units=2, num_classes=2)
    net.train(X, y, lr=0.1, epochs=200)
    preds = net.predict(X)
    assert preds == y
