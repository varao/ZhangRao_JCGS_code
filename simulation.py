import time
import sys
from path_class import *
from mh import MHsampler
from model_parameters import constructor_rate_matrix
from numpy import random
import matplotlib.pyplot as plt


def save_list(x, filename):
    import json
    f = open(filename, 'w')
    json.dump(x, f)
    f.close()


def main():
    dim, alpha, beta, sample_n, t_end = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    seed = 1
    random.seed(seed)
    var = 0.5
    t_start = 0 
    #t_end = 20
    time_breaks = list(range(1, t_end))
    mu, lamb, omega, theta = 3.0, 2.0, 5.0, 2.0
    pi_0 = [1. / dim for i in range(dim)]
    k = 2
    print('simulation starts!')
    print('random seed:', seed)
    print('alpha:', alpha)
    print('beta:', beta)
    print('dim:', dim)
    print('sample size:', sample_n)
    print('time interval: [', t_start, ', ', t_end, ']')
    print('prior: mu', 'lamb', 'omega', 'theta: ', mu, lamb, omega, theta)
    print('variance:', var)
    print('pi_0', pi_0)
    # sample observations
    rate_matrix = constructor_rate_matrix(alpha, beta, dim)
    print('rate matrix: ', rate_matrix)
    base_path = MJPpath([0], [], t_start=t_start, t_end=t_end, rate_matrix=rate_matrix, initial_pi=pi_0)
    base_path.generate_newpath()
    observation = Observation(t_start=t_start, t_end=t_end)
    observation.sample_observation(time_breaks, base_path)
    # sample alpha and beta from the posterior
    t0 = time.clock()
    samples, pos_alphas_mh, pos_betas_mh = MHsampler(observation, pi_0, sample_n, [t_start, t_end], k, mu, lamb, omega, theta, var)
    dt = time.clock() - t0
    save_list(pos_alphas_mh, "results/alpha_mh_var" + str(var))
    save_list(pos_betas_mh, "results/beta_mh_var" + str(var))
    print("run time: ", dt,  " seconds for", sample_n, "samples.")
    plt.hist(pos_alphas_mh, color='blue', label='MH-alpha', alpha=0.5)
    plt.legend(prop={'size': 20})
    plt.title("posterior alpha")
    plt.savefig('results/posterior_alpha.pdf')
    plt.close()
    plt.hist(pos_betas_mh, color='blue', label='MH-beta', alpha=0.5)
    plt.title("posterior beta")
    plt.savefig('results/posterior_beta.pdf')
    plt.close()
    return


if __name__ == '__main__':
    main()
