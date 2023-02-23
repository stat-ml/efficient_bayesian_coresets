# Efficient Bayesian Coresets

This repository contains code for the Master's thesis project on improving efficiency of Bayesian coreset construction algorithms done in 2021-2022 (Skoltech).

## Methods

The algorithms are classifed as follows
- Non-iterative: 
  - Sensitivity-based Importance Sampling **(Campbell, Broderick 2019)**
- Iterative:
  - With constraint convexification: 
    - Sensitivity-based Frank-Wolf **(Campbell, Broderick 2019)**
  - Without constraint convexification: 
    - Greedy Iterative Geodesic Approach **(Campbell, Broderick 2018)**
    - Iterative Hard Thresholding **(Zhang et al. 2021)**
    - Sparse Variational Inference **(Campbell, Beronov 2019)** 

## Implementation

`ebc` library implements sequential and parallel version of these five algorithms. Its structure is as follows.
- `ebc/approximations.py`: routines for density approximation (Random Walk Metropolis-Hastings, Laplace) and sampling (Random Kitchen Sinks).
- `ebc/gaussian.py`: utility functions for Gaussian random variables. 
- `ebc/bca.py`: implementation of `BayesianCoresetAlgorithm()`, a parent class (considered as protocol) for all five algorithms.
- `ebc/sequential/`:
  - `non_iterative.py`: implementation of `SensitivityBasedIS()` class.
  - `iterative_with_convexification.py`: implementation of `SensitivityBasedFW()` class.
  - `iterative_no_convexification.py`: implementation of `GIGA()`, `IHT()`, and `SparseVI()` class.

## Experiments



