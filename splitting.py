import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp

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

def split_randomly(x, num_proc = None):
    probs = np.ones(len(x)) / len(x)
    return distribute(probs, num_proc)