from bca import BayesianCoresetAlgorithm
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from approximations import random_kitchen_sinks

class SensitivityBasedIS(BayesianCoresetAlgorithm):

    def __init__(self, X, y = None):
        super().__init__(X, y)

    def __estimate_likelihood_gram_matrix(self, norm, norm_attributes):
        '''
        Implementation of Step 1 of Generic Algorithm
        '''
        if norm in ["uniform", "chebyshev"]:
            if norm_attributes is None:
                num_clusters = 8
                R = 1e-5
            else:
                num_clusters = norm_attributes["num_clusters"]
                R = norm_attributes["R"]

            # Cluster points with KMeans
            km = KMeans(num_clusters)
            point_labels = km.fit_predict(self.X)

            # Calculate the number of points in each cluster
            Gs = Counter(point_labels)
            Gs = [Gs[key] for key in sorted(Gs.keys(), reverse=False)]

            sense = np.zeros(self.n)

            for i in range(self.n):
                denom_sum = 0
                for p in range(len(Gs)):
                    Z = self.X[point_labels == p]
                    Z = Z[Z != self.X[i]]
                    rand_ind = np.random.randint(0, len(Z), 1)
                    bar_Z_minus_n = np.mean(Z[rand_ind])
                    denom_sum += Gs[p] * np.exp(R * np.linalg.norm(bar_Z_minus_n - self.X[i]))
                sense[i] = self.n / (1 + denom_sum)

            return np.diag(sense)

        elif norm == "2":
            if norm_attributes is None:
                raise ValueError("Cannot use 2-norm without log-likelihood function")

            log_lik = norm_attributes["log_likelihood"]
            log_lik_start = norm_attributes["log_likelihood_start_value"]
            S = norm_attributes["S"]
            approx = norm_attributes["approx"]
            MCMC_subs_size = norm_attributes["MCMC_subs_size"]
            v = random_kitchen_sinks(self.X, self.y, 1, approx, log_lik, log_lik_start, S, norm = "2", MCMC_subs_size = MCMC_subs_size)
            return v @ v.T

        elif norm == "F":
            if norm_attributes is None:
                raise ValueError("Cannot use 2-norm without log-likelihood function and its gradient")

            log_lik = norm_attributes["log_likelihood"]
            log_lik_start = norm_attributes["log_likelihood_start_value"]
            log_lik_grad = norm_attributes["log_likelihood_gradient"]
            S = norm_attributes["S"]
            approx = norm_attributes["approx"]
            MCMC_subs_size = norm_attributes["MCMC_subs_size"]
            v = random_kitchen_sinks(self.X, self.y, 1, approx, log_lik, log_lik_start, S, norm = "2", grad_log_likelihood = log_lik_grad,
                                     MCMC_subs_size = MCMC_subs_size)
            return v @ v.T

    def __estimate_directions(self, likelihood_gram_matrix):
        '''
        Implementation of Step 2 of Generic Algorithm
        '''
        sensitivities = np.diag(likelihood_gram_matrix).reshape(-1, 1)
        return sensitivities


    def __choose_next_index(self):
        '''
        Implementation of Step 3 of Generic Algorithm
        '''
        raise NotImplementedError

    def __update_weights(self, sensitivities, k):
        '''
        Implementation of Step 4 of Generic Algorithm
        '''
        M = np.random.multinomial(k, sensitivities[:, 0].tolist() / np.sum(sensitivities), 1).reshape(-1, 1)
        return (np.sum(sensitivities) / sensitivities * M / k).reshape(-1, 1)

    def run(self, k = 10, likelihood_gram_matrix = None, norm = "uniform", norm_attributes = None):
        '''
        Parameters
        ----------
        k: int
            Maximum coreset size.

        likelihood_gram_matrix: array_like
            Array of shape (n_samples, n_samples) with the Gram matrix of likelihood vectors.
            If unavailable, will be estimated.

        norm_attributes: dict
            Dictionary with norm parameters. 
            Uniform norm: {"num_clusters": ..., "R": ...}
        '''

        # Step 1
        if likelihood_gram_matrix is None:
            if norm.lower() not in ["uniform", "chebyshev", "2", "f"]:
                raise NotImplementedError
            else:
                likelihood_gram_matrix = self.__estimate_likelihood_gram_matrix(norm, norm_attributes)

        # Step 2
        sensitivities = self.__estimate_directions(likelihood_gram_matrix)

        # Step 3 is excluded in this algorithm
        # pass

        # Step 4
        self.w = self.__update_weights(sensitivities, k)
        self.I = np.arange(self.n)[self.w[:, 0] > 0].tolist()

        return self.w, self.I