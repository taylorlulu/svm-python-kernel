import numpy as np

class Zscore:
    def __init__(self, x):
        if x.ndim == 3:
            x = np.reshape(x, [-1, x.shape[2]])
        m = x.shape[0]
        self.mu = np.mean(x, axis = 0)
        self.sigma = np.std(x, axis = 0)

    def zscore_sample(self, x):
        tmp = x
        if x.ndim == 3:
            tmp = np.reshape(x, [-1, x.shape[2]])
        m = tmp.shape[0]
        mu = np.tile(self.mu, [m, 1])
        sigma = np.tile(self.sigma, [m, 1])
        z = (tmp - mu) / sigma
        if x.ndim == 3:
            return np.reshape(z, (x.shape))
        return z