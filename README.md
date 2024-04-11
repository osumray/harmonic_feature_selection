# Quiver Laplacians and Feature Selection

Code produced for the paper Quiver Laplacians and Feature Selection[^1]. 
## Use
[topology.py](./harmonic_feature_selection/topology.py) contains classes for constructing simplicial complexes and covers.
The `Cover` class takes as input a numpy array of data points (rows are data points), and an `OrderedSet` of `frozenset`s of indices of the rows of the data points.
The `Nerve` class constructs a simplicial complex from a `Cover` object.

[harmonic.py](./harmonic_feature_selection/harmonic.py) contains classes to compute compatible features.
`OriginalFeatureSelectionProblem` and `DualFeatureSelectionProblem` correspond to the constructs in Theorem 7.1 and Theorem 7.2 of [^1] respectively.
They both take: a `Simplicial Complex` and a dictionary of points associated to each simplex; a [kernel](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process.kernels).
`OriginalFeatureSelectionProblem` also takes a particular simplex to reduce the problem onto.


[^1]: [Quiver Laplacians and Feature Selection](https://arxiv.org/abs/2404.06993), Otto Sumray, Heather A. Harrington, Vidit Nanda, 2024.
