import numpy as np


class BayesianCoresetAlgorithm():

    def __init__(self, X, y = None):
        """Initialize a BCA instance.
        This is a generic template for all Bayesian coreset algorithms and should not be used on its own.

        Parameters
        ----------
        X : array_like
        Array of shape (n_samples, n_features).
        
        y : array_like (optional, default = None)
        Array of shape (n_samples,).
        """
        # Store data
        self.X = X
        self.y = y

        # Store number of datapoints (n) and features (k)
        self.n = X.shape[0]
        self.d = X.shape[1]

        # Store weights as array
        self.w = np.zeros(self.n).reshape(-1, 1)

        # Store coreset indicies as a list
        self.I = []

    def __estimate_likelihood_gram_matrix(self):
        '''
        Implementation of Step 1 of Generic Algorithm
        '''
        pass

    def __estimate_directions(self):
        '''
        Implementation of Step 2 of Generic Algorithm
        '''
        pass

    def __choose_next_index(self):
        '''
        Implementation of Step 3 of Generic Algorithm
        '''
        pass

    def __update_weights(self):
        '''
        Implementation of Step 4 of Generic Algorithm
        '''
        pass

    def run(self):
        '''
        Implementation of Generic Algorithm
        '''
        pass
