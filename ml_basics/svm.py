import numpy as np

def pegasos_optimizer(data, labels, rate, num_iter):
    data = np.array([np.append(el, [1.]) for el in data])
    weights = np.zeros(data.shape[1])
    for t in range(1, num_iter):
        i = np.random.randint(len(data))
        eta = 1.0 / (rate * t)
        if labels[i] * (weights * data[i]).sum() < 1.0:
            weights = (1.0 - 1.0 / t) * weights + eta * data[i] * labels[i]
        else:
            weights = (1.0 - 1.0 / t) * weights
        tmp = np.array([1.0, 1 / np.sqrt(rate) / np.linalg.norm(weights)])
        weights = weights * tmp.min()
    return weights
