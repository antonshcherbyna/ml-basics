import numpy as np

class NeuralClassifier:
    def __init__(self, num_inputs, num_units, num_classes):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_classes = num_classes
        self.W1 = np.random.randn(self.num_inputs, self.num_units) / np.sqrt(self.num_inputs)
        self.b1 = np.zeros((1, self.num_units))
        self.W2 = np.random.randn(self.num_units, self.num_classes) / np.sqrt(self.num_units)
        self.b2 = np.zeros((1, self.num_classes))

    def train(self, x, y, lr, num_epoch):
        for _ in range(num_epoch):
            z1 = x.dot(self.W1) + self.b1
            z1 = np.tanh(z1)
            z2 = z1.dot(self.W2) + self.b2
            y_hat = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

            delta = y_hat
            delta[range(len(x)), y] -= 1
            dW2 = z1.T.dot(delta)
            db2 = np.sum(delta, axis=0, keepdims=True)
            delta = delta.dot(self.W2.T) * (1 - np.power(z1, 2))
            dW1 = x.T.dot(delta)
            db1 = np.sum(delta, axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def predict(self, x):
        z1 = x.dot(self.W1) + self.b1
        z1 = np.tanh(z1)
        z2 = z1.dot(self.W2) + self.b2
        y_hat = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
        return np.argmax(y_hat, axis=1)
