# ml-basics

This repository contains minimal implementations of several machine learning algorithms.
The original notebooks remain in the `notebooks/` directory. A small Python
package `ml_basics` exposes the same ideas in a reusable form. Tests are located
under `tests/` and a simple `benchmark.py` script compares these
implementations with scikit-learn (if installed).

Implemented algorithms:
- Linear and logistic regression
- K-Means clustering
- Gaussian Naive Bayes
- Gaussian Discriminant Analysis
- PCA (using power iteration)
- Linear SVM with Pegasos
- A simple two-layer neural network
