import numpy as np

def fisher_norm_under_true_gaussian_posterior(x_i, x_j, mu, n):
    '''Fisher norm under gaussian posterior with mean mu.
    
    Parameters
    ----------
    x_i, x_j: array_like
        Arrays of shape (d, 1), vectors to calculate scalar product between.
    mu: array_like
        Array of shape(d, 1), vector of expectations.
    n: int
        Number of observations.
    '''
    return 2 / (1 + n) + (mu - x_i).T @ (mu - x_j)

def gaussian_multivariate_log_likelihood(x, mu, sigma):
    return -1/2 * (np.linalg.slogdet(sigma)[1] + (x - mu).T @ np.linalg.inv(sigma) @ (x-mu) + mu.shape[0] * np.log(2 * np.pi))

def gaussian_KL(sigma_p, sigma_q, mu_p, mu_q):
    # https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    KL = 1/2 * (np.linalg.slogdet(sigma_q)[1] - np.linalg.slogdet(sigma_p)[1] - mu_p.shape[0] +
                  (mu_p - mu_q).T @ np.linalg.inv(sigma_q) @ (mu_p - mu_q) +
                  np.trace(np.linalg.inv(sigma_q) @ sigma_p))
    return KL[0][0]