import numpy as np

# Numeric gradient approximation

def gradient(func, x, h=0.001):
    I = np.eye(len(x))
    return np.array([(func(x + h * _I) - func(x - h * _I)) / (2.0 * h) for _I in I])


class GradientDescent:
    def __init__(self, func, x, rate, grad=gradient, theta=0.5, delta=0.5, eps=0.001):
        self.func = func
        self.x = x
        self.rate = rate
        self.grad = grad
        self.theta = theta
        self.delta = delta
        self.eps = eps

    def minimize(self):
        iteration = 0
        while True:
            grad = self.grad(self.func, self.x)
            next_x = self.x - self.rate * grad
            rate = self.rate
            while self.func(next_x) - self.func(self.x) > -self.theta * rate * np.linalg.norm(grad) ** 2:
                rate = rate * self.delta
                next_x = self.x - rate * grad
            args_error = np.linalg.norm(np.array(self.x) - np.array(next_x)) < self.eps
            func_error = abs(self.func(next_x) - self.func(self.x)) < self.eps
            grad_error = np.linalg.norm(gradient(self.func, next_x)) < self.eps
            self.x = next_x
            iteration += 1
            if args_error and func_error and grad_error:
                return {"x": self.x, "f(x)": self.func(self.x), "iter": iteration}
            elif iteration == 10000:
                print("max iter reached, try to change parameters")
                return None


class PCA:
    def __init__(self):
        self.reduction_mat = None

    def fit(self, data, num_components):
        data = np.array(data)
        data = data - data.mean(axis=0)
        eig_vecs = np.linalg.svd(np.cov(data, rowvar=False))[0]
        self.reduction_mat = eig_vecs[:, :num_components]

    def transform(self, x):
        transformed_x = [np.matmul(self.reduction_mat.T, _x) for _x in x]
        return np.array(transformed_x)


class AdamRegressor:
    def __init__(self, regr_type, l2=False, l2_rate=0.01, rate=0.1, beta1=0.9, beta2=0.999, zero_correction=1e-8):
        self.regr_type = regr_type
        self.l2 = l2
        self.l2_rate = l2_rate
        self.rate = rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.zero_correction = zero_correction
        self.theta = None
        self.timestep = 0
        self.moment1 = 0
        self.moment2 = 0

    def gradient(self, x, y):
        x = np.append(x, [1])
        if self.regr_type == "linear":
            tmp = (x * self.theta).sum() - y
        elif self.regr_type == "logistic":
            tmp = 1.0 / (1.0 + np.exp(-(x * self.theta).sum())) - y
        grad = x * tmp
        if self.l2:
            grad = grad + self.l2_rate * self.theta
        return np.array(grad)

    def train_step(self, batch_x, batch_y):
        self.timestep += 1
        grads = []
        for x, y in zip(batch_x, batch_y):
            grads.append(self.gradient(x, y))
        grad = np.sum(grads, axis=0) / len(grads)
        self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * grad
        self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * grad ** 2
        timestep_rate = self.rate * np.sqrt(1 - self.beta2 ** self.timestep) / (1 - self.beta1 ** self.timestep)
        self.theta = self.theta - timestep_rate * self.moment1 / (np.sqrt(self.moment2) + self.zero_correction)

    def fit(self, x, y, batch_size, epoch_num):
        if hasattr(x[0], "__len__"):
            feats_size = len(x[0])
        else:
            feats_size = 1
        num_batches = len(x) // batch_size
        x = np.reshape(x[: num_batches * batch_size], [num_batches, batch_size, feats_size])
        y = np.reshape(y[: num_batches * batch_size], [num_batches, batch_size])
        self.theta = np.random.random(feats_size + 1)
        for _ in range(epoch_num):
            for batch_x, batch_y in zip(x, y):
                self.train_step(batch_x, batch_y)

    def predict(self, x):
        x = np.array([np.append(_x, [1]) for _x in x])
        if self.regr_type == "linear":
            prediction = np.array([(_x * self.theta).sum() for _x in x])
        elif self.regr_type == "logistic":
            probs = np.array([1.0 / (1.0 + np.exp(-(_x * self.theta).sum())) for _x in x])
            probs = np.array([[prob, 1 - prob] for prob in probs])
            prediction = [probs, np.array([prob.argmax() for prob in probs])]
        return prediction
