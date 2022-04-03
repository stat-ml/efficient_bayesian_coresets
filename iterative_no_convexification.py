from asyncio.events import BaseDefaultEventLoopPolicy
from bca import BayesianCoresetAlgorithm
import numpy as np
from approximations import RWMH, random_kitchen_sinks

class SparseVI(BayesianCoresetAlgorithm):

    def __init__(self, X, y=None):
        super().__init__(X, y)


    def __estimate_likelihood_gram_matrix(self, norm_attributes):
        '''
        Implementation of Step 1 of Generic Algorithm
        '''
        log_post = norm_attributes["log_posterior"]
        log_lik = norm_attributes["log_likelihood"]
        log_post_start = norm_attributes["log_posterior_start_value"]
        S = norm_attributes["S"]
        approx = norm_attributes["approx"]

        if approx == "Laplace":
            pass
        elif approx == "MCMC":
            samples = RWMH(self.X, self.y, self.w, S * 2, log_post, log_post_start, S)

            # Compute the N-dimensional potential vector for each sample
            # shape(fns) = (n, S)
            fns = np.zeros((self.n, S))
            for a in range(S):
                fns[:, a] = log_lik(samples[:, a].reshape(-1, 1), self.X, self.y, 1).reshape(-1, 1)[:, 0]
            # Subtract mean
            gns = fns - np.mean(fns, axis = 1).reshape(-1, 1)

            # Estimate correlations between the potentials and the residual error
            mean_gs_gst = np.zeros((self.n, self.n))
            
            for s in range(S):
                gs = gns[:, s].reshape(-1, 1)
                mean_gs_gst += gs @ gs.T
                
            mean_gs_gst /= S # n x n

            return mean_gs_gst

        else:
            raise NotImplementedError

        

    def __estimate_directions(self, likelihood_gram_matrix):
        '''
        Implementation of Step 2 of Generic Algorithm
        '''
        var = np.sqrt(np.diag(likelihood_gram_matrix))
        mean_gs_gst_w = likelihood_gram_matrix @ (1 - self.w)

        # For numerical stability
        var[var < 1e-05] = 1e-05

        var = var.reshape(-1, 1)
        corr = mean_gs_gst_w / var # in RN

        return corr

    def __choose_next_index(self, correlations):
        '''
        Implementation of Step 3 of Generic Algorithm
        '''
        return np.argmax(correlations)

    def __update_weights(self):
        '''
        Implementation of Step 4 of Generic Algorithm
        '''
        pass

    def run(self, k = 10, likelihood_gram_matrix = None, norm_attributes = None, T = 30, gamma_func = None):
        '''
        Implementation of Generic Algorithm
        '''

        if gamma_func is None:
            gamma_func = lambda x: 1 / x

        # Step 1
        if likelihood_gram_matrix is None:
            likelihood_gram_matrix = self.__estimate_likelihood_gram_matrix(norm_attributes)

        for _ in range(k):
            # Step 2
            correlations = self.__estimate_directions(likelihood_gram_matrix).reshape(-1, 1)

            # Step 3
            next_best_ind = self.__choose_next_index(correlations)
            if next_best_ind not in self.I:
                self.I.append(next_best_ind)

            # Step 4
            for t in range(1, T + 1):
                if likelihood_gram_matrix is None:
                    likelihood_gram_matrix = self.__estimate_likelihood_gram_matrix(norm_attributes)

                correlations = -self.__estimate_directions(likelihood_gram_matrix).reshape(-1, 1)

                indic = np.zeros(self.n)
                indic[self.I] = 1
                self.w = self.w - gamma_func(t) * correlations * indic.reshape(-1, 1)

                # For numerical stability
                self.w[self.w < 0] = 0

        return self.w, self.I

class GIGA(BayesianCoresetAlgorithm):

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

            return v.reshape(self.n, -1)

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

            return v.reshape(self.n, -1)

    def __estimate_directions(self, likelihood_vectors, ell_w):
        '''
        Implementation of Step 2 of Generic Algorithm
        '''
        ell = likelihood_vectors.sum(axis = 0).reshape(-1, 1)
        ell = ell / np.linalg.norm(ell)

        numerator = (ell - np.sum(ell * ell_w) * ell_w).reshape(-1, 1)
        dt = (numerator / np.linalg.norm(numerator)).reshape(-1, 1)
        
        norm_ell_n = np.linalg.norm(likelihood_vectors, axis = 1).reshape(-1, 1)
        likelihood_vectors = likelihood_vectors / (norm_ell_n + 1e-8)
        likelihood_vectors[norm_ell_n.flatten() == 0] = np.zeros(likelihood_vectors.shape[1])

        ell_w_matrix = np.tile(ell_w, self.n).reshape(self.n, -1)

        numerator = likelihood_vectors - likelihood_vectors @ ell_w * ell_w_matrix
        dtn = (numerator / (np.linalg.norm(numerator, axis = 1) + 1e-8).reshape(-1, 1)).reshape(self.n, -1)
        dtn[np.linalg.norm(numerator, axis = 1) == 0] = np.zeros(likelihood_vectors.shape[1])

        return dt, dtn

    def __choose_next_index(self, dt, dtn):
        '''
        Implementation of Step 3 of Generic Algorithm
        '''
        return np.argmax(dtn @ dt)

    def __update_weights(self, likelihood_vectors, next_best_index, ell_w):
        '''
        Implementation of Step 4 of Generic Algorithm
        '''
        norm_ell = np.linalg.norm(likelihood_vectors.sum(axis = 0))
        ell = (likelihood_vectors.sum(axis = 0) / norm_ell).reshape(-1, 1)

        norm_ell_n = np.linalg.norm(likelihood_vectors, axis = 1).reshape(-1, 1)
        likelihood_vectors = likelihood_vectors / norm_ell_n

        zeta0 = np.sum(ell * likelihood_vectors[next_best_index, :].reshape(-1, 1))
        zeta1 = np.sum(ell * ell_w.reshape(-1, 1))
        zeta2 = np.sum(ell_w.reshape(-1, 1) * likelihood_vectors[next_best_index, :].reshape(-1, 1))

        # Compute the step size
        numerator = zeta0 - zeta1 * zeta2
        denominator = zeta0 - zeta1 * zeta2 + zeta1 - zeta0 * zeta2
        gamma = numerator / denominator

        if gamma < 0:
            gamma = 0
        elif gamma > 1:
            gamma = 1

        numerator = ((1-gamma) * ell_w.reshape(-1, 1) + gamma * likelihood_vectors[next_best_index, :].reshape(-1, 1)).reshape(-1, 1)
        new_l_w = (numerator / np.linalg.norm(numerator)).reshape(-1, 1)

        indic = np.zeros(self.n)
        indic[next_best_index] = 1
        numerator2 = ((1-gamma) * self.w + (gamma * indic).reshape(-1, 1)).reshape(-1, 1)
        new_w = (numerator2 / np.linalg.norm(numerator2)).reshape(-1, 1)

        return new_l_w, new_w

    def run(self, k = 10, likelihood_vectors = None, norm = "2", norm_attributes = None):
        '''
        Implementation of Generic Algorithm
        '''
        # Step 1
        if likelihood_vectors is None:
            likelihood_vectors = self.__estimate_likelihood_gram_matrix(norm, norm_attributes)
        
        ell_w = np.zeros(likelihood_vectors.shape[1]).reshape(-1, 1)
        
        for _ in range(k):
            # Step 2
            dt, dtn = self.__estimate_directions(likelihood_vectors, ell_w)

            # Step 3
            next_best_index = self.__choose_next_index(dt, dtn)

            if next_best_index not in self.I:
                self.I.append(next_best_index)

            # Step 4
            ell_w, self.w = self.__update_weights(likelihood_vectors, next_best_index, ell_w)

        # Step 4
        # Scale the weights optimally
        norm_ell = np.linalg.norm(likelihood_vectors.sum(axis = 0))
        ell = (likelihood_vectors.sum(axis = 0) / norm_ell).reshape(-1, 1)
        norm_ell_n = np.linalg.norm(likelihood_vectors, axis = 1).reshape(-1, 1)

        self.w = self.w * norm_ell / norm_ell_n * (ell_w * ell).sum()

        return self.w, self.I
        

class IHT(BayesianCoresetAlgorithm):

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
            return v.reshape(self.n, -1)

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
            return v.reshape(self.n, -1)

    def __estimate_directions(self, likelihood_vectors, z):
        '''
        Implementation of Step 2 of Generic Algorithm
        '''
        phi = likelihood_vectors.reshape(-1, self.n)
        y = np.sum(likelihood_vectors, axis = 0).reshape(-1, 1)
        return -2 * phi.T @ (y - phi @ z)

    def __choose_next_index(self, gradients, k):
        '''
        Implementation of Step 3 of Generic Algorithm
        '''
        all_indexes = set(np.arange(len(gradients.flatten())).tolist())
        z_indexes = set(self.I)
        proj_indexes = list(all_indexes - z_indexes)

        to_project = gradients.flatten()[proj_indexes]
        ind_in_to_project = np.argsort(to_project)[-k:].tolist()
        ind_in_proj_indexes = np.array(proj_indexes)[ind_in_to_project]
        total_indeces = list(z_indexes.union(ind_in_proj_indexes))

        upd_grad = np.zeros_like(gradients)
        upd_grad[total_indeces] = gradients[total_indeces]

        return upd_grad

    def __update_weights(self, gradients, tilde_gradients, likelihood_vectors, z, k):
        '''
        Implementation of Step 4 of Generic Algorithm
        '''
        phi = likelihood_vectors.reshape(-1, self.n)
        y = np.sum(likelihood_vectors, axis = 0).reshape(-1, 1)

        mu = np.square(np.linalg.norm(tilde_gradients)) / (2 * np.square(np.linalg.norm(phi @ tilde_gradients)))

        # Projection
        w_new = z - mu * gradients
        w_new[w_new < 0] = 0
        n_k_smallest = np.argsort(w_new.flatten())[:(self.n - k)].tolist()
        w_new[n_k_smallest] = 0
        w_new = w_new.reshape(-1, 1)

        tau = np.sum((y - phi @ w_new) * (phi @ (w_new - self.w))) / (2 * np.square(np.linalg.norm(phi @ (w_new - self.w))))
        z_new = w_new + tau * (w_new - self.w)

        return w_new, z_new

    def run(self, k = 10, likelihood_vectors = None, norm = "2", norm_attributes = None):
        '''
        Implementation of Generic Algorithm
        '''
        # Step 1
        if likelihood_vectors is None:
            if norm.lower() not in ["2", "f"]:
                raise NotImplementedError
            else:
                likelihood_vectors = self.__estimate_likelihood_gram_matrix(norm, norm_attributes)

        # Initialization
        z = np.zeros_like(self.w).reshape(-1, 1)

        for iter in range(1000):
            # Step 2
            gradients = self.__estimate_directions(likelihood_vectors, z)

            # Step3
            tilde_gradients = self.__choose_next_index(gradients, k)

            # Step 4
            w_new, z = self.__update_weights(gradients, tilde_gradients, likelihood_vectors, z, k)

            if np.linalg.norm(w_new - self.w) <= np.linalg.norm(self.w) * 1e-5:
                self.w = w_new
                break
            
            self.w = w_new
            self.I = np.arange(self.n)[z.flatten() > 0].tolist()
            
        self.I = np.arange(self.n)[self.w.flatten() > 0].tolist()

        return self.w, self.I