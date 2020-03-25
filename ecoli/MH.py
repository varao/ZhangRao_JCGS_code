## This one is the latest version DEC 11 2017
import numpy as np
from GBS import SampleVirtualJumps
from GBS import CalLikelihood
from FFBS import *


# parameters = [alpha, beta]
def MHUpdate(observation, path_old, k, lambs, alpha, beta, priors, cov):
    # basic information
    a1, b1 = priors[0]
    a2, b2 = priors[1]
    c1, d1 = priors[2]
    c2, d2 = priors[3]
    initial_pi = copy.deepcopy(path_old.initial_pi)
    N = len(initial_pi)
    t_start = path_old.t_start
    t_end = path_old.t_end
    w = len(path_old.T)
    # copy old parameters
    matrix_old = copy.deepcopy(path_old.rate_matrix)
    alpha_old = alpha
    beta_old = beta
    # Step 1 Propose a theta* based on log normal distribution with variance var
    alpha_new, beta_new, lamb1_new, lamb2_new = propose([alpha_old, beta_old, lambs[0], lambs[1]], cov)
    lambs_new = [lamb1_new, lamb2_new]
    matrix_new = constructor_rate_matrix(alpha_new, beta_new)
    OMEGA_old = get_omega(matrix_old, k / 2.)
    OMEGA_new = get_omega(matrix_new, k / 2.)
    OMEGA = OMEGA_new + OMEGA_old
    # Step 2 Sample W*:
    uipath_old = SampleVirtualJumps(path_old, OMEGA)
    # calculate likelihood
    likelihood_old = CalLikelihood(observation, uipath_old.T, [t_start, t_end], lambs)
    # Forward calculate P(Y | W, alpha_old, beta_old)
    logp_old,  ALPHA_old = FF(likelihood_old, initial_pi, OMEGA, uipath_old)
    # Temperarily change the rate matrix in order to cal marginal probability
    uipath_old.rate_matrix = matrix_new
    # likelihood_new = likelihood_old
    likelihood_new = CalLikelihood(observation, uipath_old.T, [t_start, t_end], lambs_new)
    # Forward calculate P(Y | W, alpha_old, beta_new, lamb1_new, lamb2_new)
    logp_new,  ALPHA_new = FF(likelihood_new, initial_pi, OMEGA, uipath_old)
    # Step 3 decide whether exchange theta* and theta
    accept_rate = logp_new - logp_old + (a1 - 1) * (np.log(alpha_new) - np.log(alpha_old)) - b1 * (alpha_new - alpha_old) + (a2 - 1) * (np.log(beta_new) - np.log(beta_old)) - b2 * (beta_new - beta_old)
    accept_rate += (c1 - 1) * (np.log(lambs_new[0]) - np.log(lambs[0])) - d1 * (lambs_new[0] - lambs[0]) + (c2 - 1) * (np.log(lambs_new[1]) - np.log(lambs[1])) - d2 * (lambs_new[1] - lambs[1])
    accept_rate = min(0, accept_rate)
    if np.log(np.random.uniform()) < accept_rate:
        # proposed beta is accepted
        beta_old = beta_new
        alpha_old = alpha_new
        lambs = lambs_new
        path_new = BS(initial_pi, ALPHA_new, likelihood_new, OMEGA, uipath_old)
    else:
        # rejected
        uipath_old.rate_matrix = matrix_old
        path_new = BS(initial_pi, ALPHA_old, likelihood_old, OMEGA, uipath_old)
    # Step 4 Delete virtual jumps
    path_new.delete_virtual()
    return path_new, alpha_old, beta_old, lambs

def MHsampler(observation, pi_0, sample_n, T_interval, k, priors, cov):
    alpha_list = []
    beta_list = []
    lamb1_list = []
    lamb2_list = []
    path_list = []
    alpha_old = 0.05
    beta_old = 0.71
    lambs_old = [0.027, 0.495]
    rate_matrix = constructor_rate_matrix(alpha_old, beta_old)
    path_old = MJPpath(S=[0], T=[], t_start=T_interval[0], t_end=T_interval[1], rate_matrix=rate_matrix, initial_pi=pi_0)
    path_old.generate_newpath()
    path_list.append(copy.deepcopy(path_old))
    for i in range(sample_n):
        if i == sample_n / 2:
            print("MH 50%")
        if i == sample_n - 1:
            print("MH 100%")
        path_new, alpha_new, beta_new, lambs_new = MHUpdate(observation, path_old, k, lambs_old, alpha_old, beta_old, priors, cov)
        path_list.append(copy.deepcopy(path_new))
        alpha_list.append(alpha_new)
        beta_list.append(beta_new)
        lamb1_list.append(lambs_new[0])
        lamb2_list.append(lambs_new[1])
        path_old, alpha_old, beta_old, lambs_old = path_new, alpha_new, beta_new, lambs_new
    return alpha_list, beta_list, lamb1_list, lamb2_list
