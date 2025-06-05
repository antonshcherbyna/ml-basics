"""Expose basic machine-learning algorithms as a small reusable package.

All modules were extracted from the companion notebooks and grouped under the
``ml_basics`` package so they can be imported in regular Python scripts.
"""

from .gda import GaussianDiscriminant
from .kmeans import PCA, KMeans
from .naive_bayes import NaiveBayes
from .neural_network import NeuralClassifier
from .svm import pegasos_optimizer
from .optimization import gradient, GradientDescent, AdamRegressor

__all__ = [
    'GaussianDiscriminant',
    'PCA', 'KMeans',
    'NaiveBayes',
    'NeuralClassifier',
    'pegasos_optimizer',
    'gradient', 'GradientDescent', 'AdamRegressor'
]
