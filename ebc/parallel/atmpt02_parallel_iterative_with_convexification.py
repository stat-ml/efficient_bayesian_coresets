from ebc.bca import BayesianCoresetAlgorithm
import numpy as np
from ebc.approximations import random_kitchen_sinks
import multiprocessing as mp
from multiprocessing import get_context

def update_weights(w, n, likelihood_gram_matrix, next_best_ind):
        '''
        Implementation of Step 4 of Generic Algorithm
        '''
        # Closed-form line search for step size
        sensitivities = np.sqrt(np.diag(likelihood_gram_matrix)).reshape(-1, 1)
        w = w.reshape(-1, 1)

        first_norm = likelihood_gram_matrix.sum(axis = 0)[next_best_ind] * np.sum(sensitivities) / sensitivities[next_best_ind]
        second_norm = np.sum(likelihood_gram_matrix @ w)
        third_norm = (likelihood_gram_matrix @ w)[next_best_ind] * np.sum(sensitivities) / sensitivities[next_best_ind]
        fourth_norm = w.T @ likelihood_gram_matrix @ w

        numerator = float(first_norm) - float(second_norm) - float(third_norm) + float(fourth_norm)

        first_norm = likelihood_gram_matrix[next_best_ind, next_best_ind] * (np.sum(sensitivities) / sensitivities[next_best_ind]) ** 2
        denominator = float(first_norm) - 2 * float(third_norm) + float(fourth_norm)

        gamma = numerator / denominator

        # Add / reweight datapoints in coreset
        indic = np.zeros(n).reshape(-1, 1)
        indic[next_best_ind] = 1

        new_w = (1 - gamma) * w + gamma * np.sum(sensitivities) / sensitivities[next_best_ind] * indic

        return new_w[next_best_ind]

class ThreeParallelSensitivityBasedFW(BayesianCoresetAlgorithm):

    def __init__(self, X, y=None):
        super().__init__(X, y)
        self.__update_weights = update_weights
        

    def __estimate_likelihood_gram_matrix(self, norm, norm_attributes):
        '''
        Implementation of Step 1 of Generic Algorithm
        '''
        if norm == "2":
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
        sensitivities = np.sqrt(np.diag(likelihood_gram_matrix)).reshape(-1, 1)
        first_norm = likelihood_gram_matrix.sum(axis = 0).reshape(-1, 1) / sensitivities
        second_norm = likelihood_gram_matrix @ self.w / sensitivities

        return (first_norm - second_norm).reshape(-1, 1)

    def __choose_next_index(self, directions):
        '''
        Implementation of Step 3 of Generic Algorithm
        '''
        return np.argsort(directions.flatten())[::-1][:mp.cpu_count()].tolist()

    def run(self, k = 10, likelihood_gram_matrix = None, norm = "2", norm_attributes = None):
        '''
        Implementation of Generic Algorithm
        '''
        # Step 1
        if likelihood_gram_matrix is None:
            if norm.lower() not in ["2", "f"]:
                raise NotImplementedError
            else:
                likelihood_gram_matrix = self.__estimate_likelihood_gram_matrix(norm, norm_attributes)

        # Initialization
        sensitivities = np.sqrt(np.diag(likelihood_gram_matrix)).reshape(-1, 1)

        # Greedy initial vertex selection
        directions = likelihood_gram_matrix.sum(axis = 0).reshape(-1, 1) / sensitivities
        next_best_ind = np.argmax(directions)
        self.I.append(next_best_ind)

        # Initialize w with full weight on f
        indic = np.zeros(self.n).reshape(-1, 1)
        indic[next_best_ind] = 1
        self.w = (np.sum(sensitivities) / sensitivities * indic).reshape(-1, 1)

        # Get the number of chunks to parallelize
        if k % mp.cpu_count() == 0:
            chunks = int((k-1) // mp.cpu_count())
        else:
            chunks = int((k-1) // mp.cpu_count()) + 1

        for _ in range(chunks):
            # Step 2
            directions = self.__estimate_directions(likelihood_gram_matrix)

            # Step 3
            next_best_inds = self.__choose_next_index(directions)

            for ind in next_best_inds:
                if ind not in self.I:
                    self.I.append(ind)

            # Step 4
            pool = get_context("fork").Pool(mp.cpu_count())
            output = [pool.apply(self.__update_weights, args = [self.w.reshape(-1, 1), self.n, likelihood_gram_matrix, ind]) for ind in next_best_inds]
            pool.close()
            #self.w = np.concatenate(output, axis = 1).mean(axis = 1).reshape(-1, 1)
            self.w[next_best_inds] = np.array(output).flatten().reshape(-1, 1)

        return self.w, self.I