import random
import math

class NeuralClassifier:
    """Tiny two-layer neural network classifier."""

    def __init__(self, num_inputs, num_units, num_classes):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_classes = num_classes
        self.W1 = [[random.gauss(0,1)/math.sqrt(num_inputs) for _ in range(num_units)] for _ in range(num_inputs)]
        self.b1 = [0.0]*num_units
        self.W2 = [[random.gauss(0,1)/math.sqrt(num_units) for _ in range(num_classes)] for _ in range(num_units)]
        self.b2 = [0.0]*num_classes

    def _forward(self, x):
        z1 = [sum(x[j]*self.W1[j][i] for j in range(self.num_inputs)) + self.b1[i] for i in range(self.num_units)]
        a1 = [math.tanh(z) for z in z1]
        z2 = [sum(a1[j]*self.W2[j][k] for j in range(self.num_units)) + self.b2[k] for k in range(self.num_classes)]
        exps = [math.exp(z) for z in z2]
        s = sum(exps)
        probs = [e/s for e in exps]
        return a1, probs

    def train(self, X, y, lr=0.1, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                a1, probs = self._forward(xi)
                delta2 = [probs[k] - (1 if yi==k else 0) for k in range(self.num_classes)]
                dW2 = [[delta2[k]*a1[j] for k in range(self.num_classes)] for j in range(self.num_units)]
                db2 = delta2
                delta1 = [sum(delta2[k]*self.W2[j][k] for k in range(self.num_classes))*(1 - a1[j]**2) for j in range(self.num_units)]
                dW1 = [[delta1[i]*xi[j] for i in range(self.num_units)] for j in range(self.num_inputs)]
                db1 = delta1
                for j in range(self.num_inputs):
                    for i in range(self.num_units):
                        self.W1[j][i] -= lr*dW1[j][i]
                for i in range(self.num_units):
                    self.b1[i] -= lr*db1[i]
                for j in range(self.num_units):
                    for k in range(self.num_classes):
                        self.W2[j][k] -= lr*dW2[j][k]
                for k in range(self.num_classes):
                    self.b2[k] -= lr*db2[k]

    def predict(self, X):
        preds = []
        for x in X:
            _, probs = self._forward(x)
            preds.append(probs.index(max(probs)))
        return preds
