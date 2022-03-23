import numpy as np

def gaussian_KL(sigma_p, sigma_q, mu_p, mu_q):
    # https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    KL = 1/2 * (np.linalg.slogdet(sigma_q)[1] - np.linalg.slogdet(sigma_p)[1] - mu_p.shape[0] +
                  (mu_p - mu_q).T @ np.linalg.inv(sigma_q) @ (mu_p - mu_q) +
                  np.trace(np.linalg.inv(sigma_q) @ sigma_p))
    return KL[0][0]