from ebc.bca import BayesianCoresetAlgorithm
import numpy as np
from ebc.approximations import random_kitchen_sinks

class NEWSensitivityBasedFW(BayesianCoresetAlgorithm):

    def __init__(self, X, y=None):
        super().__init__(X, y)

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

    def __choose_next_index(self, directions, k):
        '''
        Implementation of Step 3 of Generic Algorithm
        '''
        return np.argsort(directions.flatten())[::-1][:k].tolist()

    def __update_weights(self, directions, likelihood_gram_matrix, next_best_ind):
        '''
        Implementation of Step 4 of Generic Algorithm
        '''
        # Closed-form line search for step size
        sensitivities = np.sqrt(np.diag(likelihood_gram_matrix)).reshape(-1, 1)

        #first_norm = likelihood_gram_matrix.sum(axis = 0)[next_best_ind] * np.sum(sensitivities) / sensitivities[next_best_ind]
        #second_norm = np.sum(likelihood_gram_matrix @ self.w)
        #third_norm = (likelihood_gram_matrix @ self.w)[next_best_ind] * np.sum(sensitivities) / sensitivities[next_best_ind]
        #fourth_norm = self.w.T @ likelihood_gram_matrix @ self.w

        #numerator = float(first_norm) - float(second_norm) - float(third_norm) + float(fourth_norm)

        #first_norm = likelihood_gram_matrix[next_best_ind, next_best_ind] * (np.sum(sensitivities) / sensitivities[next_best_ind]) ** 2
        #denominator = float(first_norm) - 2 * float(third_norm) + float(fourth_norm)
        #if denominator == 0:
        # denominator = 1e-10

        # gamma = numerator / denominator
        gamma = 0.3

        # Add / reweight datapoints in coreset
        indic = np.zeros(self.n).reshape(-1, 1)
        indic[np.array(next_best_ind).reshape(-1, 1)] = 1

        return (1 - gamma) * self.w + gamma * np.sum(sensitivities) / sensitivities * indic

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

        for _ in range(300):
            # Step 2
            directions = self.__estimate_directions(likelihood_gram_matrix)

            # Step 3
            next_best_ind = self.__choose_next_index(directions, k)

            for nbi in next_best_ind:
                if nbi not in self.I:
                    self.I.append(nbi)

            # Step 4
            self.w = self.__update_weights(directions, likelihood_gram_matrix, self.I).reshape(-1, 1)
            self.w[self.w < 0] = 0
            n_k_smallest = np.argsort(self.w.flatten())[:(self.n - k)].tolist()
            self.w[n_k_smallest] = 0
            self.w = self.w.reshape(-1, 1)

            self.I = np.arange(self.n)[self.w.flatten() > 0].tolist()

        return self.w, self.I