from path_class import *
from math import log
from model_parameters import get_omega, constructor_rate_matrix, propose
from FFBS import FF, BS, sampleUI, get_likelihood


def MH_update(observation, ST_old, k, parameters, mu, lamb, omega, theta, var):
    # basic information
    initial_pi = copy.deepcopy(ST_old.initial_pi)
    N = len(initial_pi)
    t_start = ST_old.t_start
    t_end = ST_old.t_end
    # copy old parameters
    matrix_old = copy.deepcopy(ST_old.rate_matrix)
    alpha_old = parameters[0]
    beta_old = parameters[1]
    # Step 1 Propose a theta* based on log normal distribution with variance var
    alpha_new, beta_new = propose(alpha_old, beta_old, var)
    matrix_new = constructor_rate_matrix(alpha_new, beta_new, N)
    OMEGA_old = get_omega(matrix_old, k / 2.)
    OMEGA_new = get_omega(matrix_new, k / 2.)
    #    OMEGA = max(OMEGA_new, OMEGA_old)
    OMEGA = OMEGA_new + OMEGA_old
    # Step 2 Sample W*:
    uipath_old = sampleUI(ST_old, OMEGA)
    # calculate likelihood
    likelihood = get_likelihood(observation, uipath_old.T, N, [t_start, t_end])
    # Forward calculate P(Y | W, alpha_old, beta_old)
    logp_old,  ALPHA_old = FF(likelihood, initial_pi, OMEGA, uipath_old)
    # Temporally change the rate matrix in order to cal marginal probability
    uipath_old.rate_matrix = matrix_new
    # Forward calculate P(Y | W, alpha_old, beta_new)
    logp_new,  ALPHA_new = FF(likelihood, initial_pi, OMEGA, uipath_old)
    # Step 3 decide whether exchange theta* and theta
    accept_rate = logp_new - logp_old + mu * (log(alpha_new) - log(alpha_old)) - lamb * (alpha_new - alpha_old) + omega * (log(beta_new) - log(beta_old)) - theta * (beta_new - beta_old)
    accept_rate = min(0, accept_rate)
    if log(random.uniform()) < accept_rate:
        # proposed beta is accepted
        beta_old = beta_new
        alpha_old = alpha_new
        ST_new = BS(initial_pi, ALPHA_new, likelihood, OMEGA, uipath_old)
    else:
        # rejected
        uipath_old.rate_matrix = matrix_old
        ST_new = BS(initial_pi, ALPHA_old, likelihood, OMEGA, uipath_old)
    # Step 4 Delete virtual jumps
    ST_new.delete_virtual()
    return ST_new, alpha_old, beta_old


def MHsampler(observation, pi_0, sample_n, T_interval, k, mu, lamb, omega, theta, var):
    # all the trajectories DON'T contain virtual jumps
    alpha_list = []
    beta_list = []
    ST_list = []
    alpha_old = 2.0
    beta_old = 1.5
    rate_matrix = constructor_rate_matrix(alpha_old, beta_old, len(pi_0))
    ST_old = MJPpath([0], [], t_start=T_interval[0], t_end=T_interval[1], rate_matrix=rate_matrix, initial_pi=pi_0)
    ST_old.generate_newpath()
    ST_list.append(copy.deepcopy(ST_old))
    for i in range(sample_n):
        if (i % 100)==0 :
            print(i)
        ST_new, alpha_new, beta_new= MH_update(observation, ST_old, k, [alpha_old, beta_old], mu, lamb, omega, theta, var)
        ST_list.append(copy.deepcopy(ST_new))
        alpha_list.append(alpha_new)
        beta_list.append(beta_new)
        ST_old, alpha_old, beta_old = ST_new, alpha_new, beta_new
    return ST_list, alpha_list, beta_list
