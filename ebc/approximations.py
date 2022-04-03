from scipy.optimize import minimize
import numpy as np

def RWMH(X, y = None, weights = None, S = 150, log_posterior = None, log_posterior_start_value = None, burnout = 0):
    # https://umbertopicchini.wordpress.com/2017/12/18/tips-for-coding-a-metropolis-hastings-sampler/
    # Unfold parameters
    params = np.zeros((len(log_posterior_start_value), S))
    params[:, 0] = log_posterior_start_value.reshape(-1, 1)[:, 0]
    
    for i in range(1, S):
        # Generate proposal
        prop = np.random.multivariate_normal(params[:, i-1].reshape(-1, 1)[:, 0], np.identity(len(log_posterior_start_value)) * 0.001)
        r = log_posterior(prop.reshape(-1, 1), X, y, weights) - log_posterior(params[:, i-1].reshape(-1, 1), X, y, weights)
        u = np.random.uniform(0, 1, 1)
        if np.log(u) < r:
            params[:, i] = prop
        else:
            params[:, i] = params[:, i-1]
    
    return params[:, burnout:]

def laplace_approximation(density, start_value, args):
    '''Laplace approximation to a density function.

    Parameters:
    ----------
    density: function
        Density function to optimize.
    start_value: array_like
        Start value for optimization
    args: tuple
        Additional values for density function optimization.
    
    Returns:
    ----------
    mu: array_like
        Mean of the normal approximation.
    sigma: array_like
        Variance of the normal approximation.
    '''
    # Find mode
    solution = minimize(density, start_value, args = args)
    # Return mode as mu and H^{-1} as variance
    return solution.x, solution.hess_inv


def random_kitchen_sinks(X, y = None, weights = 1, approx = "Laplace", log_likelihood = None, log_likelihood_start_value = None, 
                         S = 150, norm = "2", grad_log_likelihood = None, burnout = 0, MCMC_subs_size = 1):

    if norm not in ["2", "F"]:
        raise NotImplementedError

    if approx not in ["Laplace", "MCMC"]:
        raise NotImplementedError

    # For convenience
    n = X.shape[0]
    d = X.shape[1]

    # The function to optimize is the negative total likelihood
    def negative_total_log_lik(params, X, y, weights):
        return -np.sum(log_likelihood(params, X, y, weights))

    # For MCMC
    def total_log_lik(params, X, y, weights):
        return -negative_total_log_lik(params, X, y, weights)
    
    if approx == "Laplace":
        # Approximate via laplace approximation
        mu_approx, sigma_approx = laplace_approximation(negative_total_log_lik, log_likelihood_start_value, (X, y, weights))
        # Sample feature vectors...
        samples = np.random.multivariate_normal(mu_approx, sigma_approx, S) # S x (#mu + #sigma)
    elif approx == "MCMC":
        # Choose random subsample
        rand_ind = np.random.choice(np.arange(n), MCMC_subs_size, replace = False)
        if y:
            y = y[rand_ind]
        # Approximate via MCMC on subsample and sample feature vectors...
        samples = RWMH(X = X[rand_ind], y = y, weights = np.ones(n).reshape(-1, 1), S = S + burnout, log_posterior = total_log_lik, 
                       log_posterior_start_value = log_likelihood_start_value, burnout = burnout).reshape(S, -1)
                
    # ...and gradient dimension indices
    dj = np.random.randint(0, d, S)

    vn = np.zeros((n, S))

    if norm == "2":
        for a in range(S):
            vn[:, a] = log_likelihood(samples[a], X, y, weights)[:, 0] * np.sqrt(1 / S)
    elif norm == "F":
        for a in range(S):
            vn[:, a] = grad_log_likelihood(samples[a], X, y, weights).reshape(-1, d)[:, dj[a]] * np.sqrt(d / S)
    
    return vn

