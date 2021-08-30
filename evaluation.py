import numpy as np
import func
from scipy.stats import norm

def calc_path_prob(path, mymap, T, samples=None, S=1000):    
    if mymap.model == 'G':
        # path = np.flatnonzero(x)
        mu_sum = np.sum(mymap.mu[path])
        cov_sum = np.sum(mymap.cov[path][:, path])
        return norm.cdf(T, mu_sum, np.sqrt(cov_sum))
    else:
        if samples is None:
            samples = func.generate_samples(mymap, S)
        x = np.zeros(mymap.n_link)
        x[path] = 1
        return np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0)) / samples.shape[1]

