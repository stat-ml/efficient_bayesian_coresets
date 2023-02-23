# Efficient Bayesian Coresets

This repository contains code for the Master's thesis project on improving efficiency of Bayesian coreset construction algorithms done in 2021-2022 at Skoltech.

## Methods

The algorithms are classifed as follows
- Non-iterative: 
  - Sensitivity-based Importance Sampling **(Campbell, Broderick 2019)**.
- Iterative:
  - With constraint convexification: 
    - Sensitivity-based Frank-Wolf **(Campbell, Broderick 2019)**.
  - Without constraint convexification: 
    - Greedy Iterative Geodesic Approach **(Campbell, Broderick 2018)**.
    - Iterative Hard Thresholding **(Zhang et al. 2021)**.
    - Sparse Variational Inference **(Campbell, Beronov 2019)**.

"External" parallelization strategy refers to distributing data over workers, running a version of an algorithm at each worker, and aggregating results. "Internal" parallelization strategy refers to adapting internal structure of an algorithm to allow for parallelization of its steps. 

## Implementation

`ebc` library implements sequential version of the five algorithms. Its structure is as follows.
- `ebc/approximations.py`: routines for density approximation (Random Walk Metropolis-Hastings, Laplace) and sampling (Random Kitchen Sinks).
- `ebc/gaussian.py`: utility functions for Gaussian random variables. 
- `ebc/bca.py`: implementation of `BayesianCoresetAlgorithm()`, a parent class (considered as protocol) for all five algorithms.
- `ebc/sequential/`:
  - `non_iterative.py`: implementation of `SensitivityBasedIS()` class.
  - `iterative_with_convexification.py`: implementation of `SensitivityBasedFW()` class.
  - `iterative_no_convexification.py`: implementation of `GIGA()`, `IHT()`, and `SparseVI()` class.

The example below shows how to run the Sensitivity-based Frank-Wolf method using `ebc`.
``` python
from ebc.sequential.iterative_with_convexification import SensitivityBasedFW

# Define log likelihood
def log_likelihood(params, X, y, weights):
  mu = params[:d].reshape(-1, 1)
  sigma = np.diag(params[d:].reshape(-1, 1)[:, 0])
  return np.diag(gaussian_multivariate_log_likelihood(X.T, mu, sigma)).
reshape(-1, 1)

# Set parameters
na = {"log_likelihood": log_likelihood, 
      "log_likelihood_start_value": np.ones(2 * d), 
      "S": 500,
      "log_likelihood_gradient": grad_log_likelihood, 
      "approx": "MCMC",
      "MCMC_subs_size": 500,
      "log_posterior": log_posterior, 
      "log_posterior_start_value": np.ones(2 * d)}

# Init and run
sbfw = SensitivityBasedFW(x)

w2, I2 = sbfw.run(k = 10, likelihood_gram_matrix = likelihood_gram_matrix, norm = "2", norm_attributes = na)
```

`ebc` also contains `parallel/` directory with four modules. These are different attempts at implementing internal parallelization strategy for SBFW (was not finalized) **which are not reliable**. External strategy was implemented directly in the experiments and not as a standalone module.

## Experiments

`experiments` folder is structered as follows.
- `experiments/sequential/`: experiments with sequential versions of the algorithms.
  - `sequential_time_tests/`: version of `ebc/sequential/` with explicit time measurement of a method's steps.
  - `Time_Comparison.ipynb`: time measurements on the multivariate Gaussian mean problem.
  - `Quality_Comparison.ipynb` quality comparison on various problems.
  - `Miscellaneous_Experiments.ipynb`: various tests for algorithms' adequacy.
- `experiments/parallel_external/`: tuning, quality and time measurements for external parallelization strategy.
- `experiments/parallel_internal/`: experiments with  attempts from `ebc/parallel/`.
