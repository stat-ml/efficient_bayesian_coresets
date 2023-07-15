import numpy as np
import multiprocessing as mp
from multiprocessing import get_context

from ebc.sequential.iterative_no_convexification import SparseVI

def parallelize(alg, x, k, norm, na, distributed_indices, alg_name = "SBIS", y = None, num_proc = None):
    if num_proc is None:
        num_proc = mp.cpu_count() - 1
    pool = get_context("fork").Pool(num_proc)
    if alg_name in ["SBIS", "SBFW"]:
        if y is not None:
            output = [pool.apply(apply_algorithm, args = [alg, x[ind, :], k, norm, na, None, y[ind, :]]) for ind in distributed_indices]
        else:
            output = [pool.apply(apply_algorithm, args = [alg, x[ind, :], k, norm, na, None, None]) for ind in distributed_indices]
    elif alg_name == "SVI":
        if y is not None:
            output = [pool.apply(apply_svi, args = [alg, x[ind, :], k, na, None, y[ind, :]]) for ind in distributed_indices]
        else:
            output = [pool.apply(apply_svi, args = [alg, x[ind, :], k, na, None, None]) for ind in distributed_indices]
    pool.close()

    w = np.concatenate(output)
    return w

def apply_algorithm(alg, x = None, k = None, norm = None, na = None, likelihood_gram_matrix = None, y = None):
    if y is not None:
        inst = alg(x, y)
    else:
        inst = alg(x)
    w, I = inst.run(k = k, likelihood_gram_matrix = likelihood_gram_matrix, norm = norm, norm_attributes = na)
    return w

def apply_svi(alg, x = None, k = None, na = None, likelihood_gram_matrix = None, y = None):
    if y is not None:
        inst = alg(x, y)
    else:
        inst = alg(x)
    w, I = inst.run(k = k, likelihood_gram_matrix = likelihood_gram_matrix, norm_attributes = na)
    return w