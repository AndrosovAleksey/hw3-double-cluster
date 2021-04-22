#!/usr/bin/env python3
import numpy as np
from scipy import stats, optimize


def common_density(args, x):
    tau, mu1, sigma1, mu2, sigma2 = args
    density_1 = stats.norm.pdf(x, loc=mu1, scale=np.abs(sigma1))
    density_2 = stats.norm.pdf(x, loc=mu2, scale=np.abs(sigma2))
    result = tau * density_1 + (1 - tau) * density_2
    return result, density_1, density_2


def likelihood_function_log(args, x):
    result, density_1, density_2 = common_density(args, x)
    return -np.sum(np.log(np.abs(result)))
# Ищем максимум, то есть минимум отрицательной функции

def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    result_likelihood = optimize.minimize((likelihood_function_log, np.array
                                           ([tau, mu1, sigma1, mu2, sigma2]), 
                                           args=x,tol=rtol)
    return result_likelihood.x


def em_algorithm(args, x):
    tau, mu1, sigma1, mu2, sigma2 = args
    result, density_1, density_2 = common_density(args, x)
    theta_1, theta_2 = tau * density_1 / result, (1 - tau) * density_2 / result
    tau = (np.sum(theta_1) / x.size)
    mu1 = (np.sum(theta_1 * x) / np.sum(theta_1))
    sigma1 = (np.sqrt((np.sum(theta_1 * (x - mu1) ** 2)) / np.sum(theta_1)))
    mu2 = (np.sum(theta_2 * x) / np.sum(theta_2))
    sigma2 = (np.sqrt((np.sum(theta_2 * (x - mu2) ** 2)) / np.sum(theta_2)))
    return tau, mu1, sigma1, mu2, sigma2


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    next_args = tau, mu1, sigma1, mu2, sigma2
    while 1:
        previous_args = next_args
        next_args = em_algorithm(previous_args, x)
        if np.allclose(next_args, previous_args, rtol=rtol, atol=0):
            return np.asarray(next_args)


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, 
                      rtol=1e-5):
    pass


if __name__ == "__main__":
    pass
