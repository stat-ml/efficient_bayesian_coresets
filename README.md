# Distributed Bayesian Coresets

> Vladimir Omelyusik and Maxim Panov (2023). Distributed Bayesian Coresets. In proceedings of The 11th International Conference on
Analysis of Images, Social Networks and Texts – [2023](https://aistconf.org). 

A Bayesian coreset is a small weighted subsample of the original data which aims to preserve the full posterior. There are several algorithms for constructing a coreset,
- Non-iterative: 
  - Sensitivity-based Importance Sampling **(Campbell, Broderick [2019](https://www.jmlr.org/papers/volume20/17-613/17-613.pdf))**.
- Iterative:
  - With constraint convexification: 
    - Sensitivity-based Frank-Wolfe **(Campbell, Broderick [2019](https://www.jmlr.org/papers/volume20/17-613/17-613.pdf))**.
  - Without constraint convexification: 
    - Greedy Iterative Geodesic Approach **(Campbell, Broderick [2018](http://proceedings.mlr.press/v80/campbell18a/campbell18a.pdf))**.
    - Iterative Hard Thresholding **(Zhang et al. [2021](http://proceedings.mlr.press/v130/zhang21g/zhang21g.pdf))**.
    - Sparse Variational Inference **(Campbell, Beronov [2019](https://proceedings.neurips.cc/paper/2019/file/7bec7e63a493e2d61891b1e4051ef75a-Paper.pdf))**.

All of them require at least one round of likelihood function estimation on the full dataset.

A convenient way to speed up such construction without changing the underlying algorithms is to build parts of the coreset on separate processors in parallel. This distributed setting is well studied in the frequentist literature for the problem of preserving K-Means clustering **(Har-Peled et al. [2004](https://dl.acm.org/doi/pdf/10.1145/1007352.1007400?casa_token=o9veX8qyQgAAAAAA:bVIJKMjKT5LdNvz2aNBOztK805-tq7EVuWnRSahvl9E_w27zfJ5D0AWd-rGVYIK4psVZIP9CAh9E))**. If cluster centers are fixed, then one can arbitrarily partition the dataset into $r$ chunks and build a coreset for each chunk. The union of the resulting coresets is the coreset for the full dataset. In this strategy, cluster centers can be viewed as anchors which are sufficient to preserve the dataset's global properties even in the absence of the full data.

In the Bayesian case, **(Campbell et al. [2019](https://www.jmlr.org/papers/volume20/17-613/17-613.pdf))** formulated error upper bounds for random data partitioning. However, these bounds describe the worst-case scenario when the processors have no information of the dataset's geometry. 

We use relations between K-Means and EM algorithms to adapt the anchoring approach from the frequentist setting and propose a partitioning strategy that uses maximum likelihood estimates as anchors of the dataset’s global properties. A point $X_i$ is assigned to a processor $w$ with probability
```math
p(X_i \in w) = \frac{\left|\ell(X_i \mid \hat{\theta}_{ML})\right|}{\sum_{j} \left|\ell(X_j \mid \hat{\theta}_{ML})\right|} = \frac{\left|\ell_i(\hat{\theta}_{ML})\right|}{\sum_{j} \left|\ell_j(\hat{\theta}_{ML})\right|},
```
without replacement. A coreset construction algorithm is then set to output a coreset of size $k / W$, where $W$ is the number of workers. It is then run on each processor concurrently, and produced coresets are combined.

Since frequentist guarantees are applicable to arbitrary partitioning strategy, we expect our approach to be comparable in quality with random partitioning on datasets which are well-clustered by K-Means and superior to the latter on data with more sophisticated geometry on which K-Means fails.

## Experiments

To test our hypothesis, we compared the quality and running time of our strategy with those of random paritioning in the following experiments:
1. Multivariate Gaussian.
2. Univariate and Multivariate Gaussian mixture.
3. Classification on the [UCI ML Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

These experiments demonstrated that 
1. Our strategy is comparable in quality to random partitioning on data with simple geometry (one cluster).
2. The error of our strategy is bounded above by the error of random partitioning on datasets with complex geometry (several intersecting clusters).
3. These results, on average, hold on real data for which the true likelihood is not known.

Additionally, we found that these results hold irrespective of the number of processors.

The experiments can be replicated by running respective notebooks from the `examples/` folder. Raw data generated during our runs is located in the `data/` folder and corresponding plots are located in the `plots/` folder.

## Implementation

### Sequential

Although our experiments were conducted with the Frank-Wolfe method, `ebc` library implements sequential versions of the five aforementioned algorithms. Its structure is as follows.
- `ebc/approximations.py`: routines for density approximation (Random Walk Metropolis-Hastings, Laplace) and sampling (Random Kitchen Sinks).
- `ebc/gaussian.py`: utility functions for Gaussian random variables. 
- `ebc/bca.py`: implementation of `BayesianCoresetAlgorithm()`, a parent class (considered as protocol) for all five algorithms.
- `ebc/sequential/`:
  - `non_iterative.py`: implementation of `SensitivityBasedIS()` class.
  - `iterative_with_convexification.py`: implementation of `SensitivityBasedFW()` class.
  - `iterative_no_convexification.py`: implementation of `GIGA()`, `IHT()`, and `SparseVI()` class.

The example below shows how to run the Sensitivity-based Frank-Wolfe method using `ebc`.
``` python
from ebc.sequential.iterative_with_convexification import SensitivityBasedFW

# Define log likelihood
def log_likelihood(params, X, y, weights):
    mu = params[:d].reshape(-1, 1)
    sigma = np.diag(params[d:].reshape(-1, 1)[:, 0])
    return np.diag(gaussian_multivariate_log_likelihood(X.T, mu, sigma)).reshape(-1, 1)

def grad_log_likelihood(params, X, y, weights):
    d = X.shape[1]
    mu = params[:d].reshape(-1, 1)
    sigma = np.diag(params[d:].reshape(-1, 1)[:, 0])
    return (-np.linalg.inv(sigma) @ (X.T - mu)).reshape(-1, X.shape[1])

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

### Distributed

The partitioning strategies we tested are implemented in `splitted.py`. Parallel coreset construction with the Frank-Wolfe algorithm is is implemented in `parallelization.py`. This file also contains an example of running a different algorithm, SparseVI, in the distributed setting which can be used for generalizations. 

The example below shows how to run the distributed version of the Frank-Wolfe method using our partitioning strategy.
``` python
from ebc.sequential.iterative_with_convexification import SensitivityBasedFW
from splitting import split_based_on_ML, distribute
from parallelization import parallelize

# Define log likelihood
def log_likelihood(params, X, y, weights):
    mu = params[:d].reshape(-1, 1)
    sigma = np.diag(params[d:].reshape(-1, 1)[:, 0])
    return np.diag(gaussian_multivariate_log_likelihood(X.T, mu, sigma)).reshape(-1, 1)

def grad_log_likelihood(params, X, y, weights):
    d = X.shape[1]
    mu = params[:d].reshape(-1, 1)
    sigma = np.diag(params[d:].reshape(-1, 1)[:, 0])
    return (-np.linalg.inv(sigma) @ (X.T - mu)).reshape(-1, X.shape[1])

# Set parameters
na = {"log_likelihood": log_likelihood, 
      "log_likelihood_start_value": np.ones(2 * d), 
      "S": 500,
      "log_likelihood_gradient": grad_log_likelihood, 
      "approx": "MCMC",
      "MCMC_subs_size": 500,
      "log_posterior": log_posterior, 
      "log_posterior_start_value": np.ones(2 * d)}

# Partition
# Note: for better efficiency, in the notebooks, we maximize each likelihood explicitly
# as opposed to running general likelihood optimization
full_inds = split_based_on_ML(x)

# Run in parallel
w, _ = parallelize(alg = SensitivityBasedFW, x = x, k = int(i // mp.cpu_count()), norm = "2",
                   na = na, distributed_indices = full_inds)
```

