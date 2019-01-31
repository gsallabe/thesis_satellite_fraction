import numpy as np
from scipy.stats.distributions import norm, trapz, uniform
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

def bias_lhs_draws(samples, ranges):

    samples[:,-2:] = add_correlation(samples[:,-2:], np.array([[1, -0.8], [-0.8, 1]]))
    samples = trapz(0.3, 0.7).ppf(samples) # Bias the samples to the center
    samples = ranges[:,0] + (ranges[:,1] - ranges[:,0]) * samples

    return samples

def add_correlation(s2, corr_matrix):
    assert s2.shape[1] == 2 and corr_matrix.shape == (2, 2)

    s2 = norm().ppf(s2)

    d2 = multivariate_normal([0, 0], corr_matrix).rvs(len(s2))

    d2 = d2[np.argsort(d2[:,0])]
    s2 = s2[np.argsort(s2[:,0])]

    s2[:,1] = d2[:,1]

    return uniform().ppf(norm.cdf(s2))
