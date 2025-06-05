# ml-basics

Simple implementation of basic machine learning algorithms for educational purposes.

This repository now exposes a small Python package `ml_basics` that contains the
implementations originally provided in the notebooks. You can import the classes
and helper functions directly:

```python
from ml_basics import (
    GaussianDiscriminant, NaiveBayes, KMeans, PCA, NeuralClassifier,
    pegasos_optimizer, GradientDescent, AdamRegressor
)
```

Each module mirrors the code from the corresponding notebook so it can be used
in regular Python scripts without relying on Jupyter.
