import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
from ebc.sequential.non_iterative import SensitivityBasedIS
from sklearn.cluster import KMeans

def distribute(probs, num_proc = None):

    if num_proc is None:
        num_proc = mp.cpu_count() - 1

    N = len(probs)

    full_inds = []
    chunk_size = N // num_proc
    
    for _ in range(num_proc):
        inds = np.random.choice(np.arange(N), chunk_size, replace = False, p = probs)
        probs[inds] = 0
        probs = probs / np.sum(probs + 1e-15)
        full_inds.append(inds.flatten().tolist())

    full_inds[-1].extend(np.arange(N)[probs.flatten() != 0].tolist())

    return full_inds


def split_based_on_ML(x, x0, negative_summed_log_likelihood, log_likelihood, y = None):
    '''
    Returns:
    ----------
    full_inds: list[list]
        Lists of indices for each processor.
    '''
    # Get ML estimates
    out = minimize(negative_summed_log_likelihood, x0, (x, y, None), method = "CG")

    # Get probability estimates
    log_liks = log_likelihood(out.x, x, y, None)
    probs = np.abs(log_liks) / np.sum(np.abs(log_liks))
    probs = probs.flatten()

    return distribute(probs)

def split_based_on_sensitivities(x, norm, norm_attributes):

    sbis = SensitivityBasedIS(x)
    likelihood_gram_matrix = sbis.estimate_likelihood_gram_matrix("uniform", None)
    sensitivities = sbis.estimate_directions(likelihood_gram_matrix)
    sens = sensitivities.copy()
    probs = sens.flatten() / np.sum(sens)
    
    return distribute(probs)

def split_based_on_KMeans(x):
    km = KMeans(mp.cpu_count() - 1)
    labels = km.fit_predict(x)
    inds = [np.where(labels == i)[0].tolist() for i in range(mp.cpu_count() - 1)]
    return inds

def split_randomly(x, num_proc = None):
    probs = np.ones(len(x)) / len(x)
    return distribute(probs, num_proc)