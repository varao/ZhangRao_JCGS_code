import time
import sys
import string
import numpy as np
import pickle
import time
from MH import *
from GBS import *
from ExperimentConfig import *
import json
import matplotlib.pyplot as plt

def my_save(list, filename):
    f = open(filename, 'w')
    json.dump(list, f)
    f.close()

def parseArgs(args):
    #"""Parses arguments vector, looking for switches of the form -key {optional value}.
    #For example:
	#parseArgs([ 'template.py', '-s', '1', '-k', '1', '-n', '5000', '-bn' , '2000'}"""
    args_map = {}
    curkey = None
    for i in range(1, len(args)):
        if args[i][0] == '-':
            args_map[args[i]] = True
            curkey = args[i]
        else:
            assert curkey
            args_map[curkey] = args[i]
            curkey = None
    return args_map

def validateInput(args):
    args_map = parseArgs(args)
    if '-s' in args_map:
        seed = int(args_map['-s'])
    if '-k' in args_map:
        kappa = float(args_map['-k'])
    if '-n' in args_map:
        n_sample = int(args_map['-n'])
    if '-bn' in args_map:
        n_burnin = int(args_map['-bn'])
    return seed, kappa, n_sample, n_burnin

def experiment(seedIndex, kappa, pi_0, sample_n, T_interval, k, priors, burnin_size):
    """
    :param seedIndex:
    :param kappa:
    :param pi_0:
    :param sample_n:
    :param T_interval:
    :param k:
    :param priors:
    :param kappa:
    :param burnin_size:
    :return:
    """
    np.random.seed(seedIndex)
    with open('lag_inner', 'rb') as f:
        observation = pickle.load(f)
        f.close()
    print("First Step GBS starts!")
    burnin_alpha_list, burnin_beta_list, burnin_lamb1_list, burnin_lamb2_list = GBSsampler(observation, pi_0, burnin_size, T_interval, k, priors)
    SimulationIndex = seedIndex
    X = np.stack((burnin_alpha_list, burnin_beta_list, burnin_lamb1_list, burnin_lamb2_list), axis=0)
    covariance = np.cov(X)
    cov = covariance * kappa
    print("Second Step MH starts!")
    alpha_list, beta_list, lamb1_list, lamb2_list = MHsampler(observation, pi_0, sample_n, T_interval, k, priors, cov)
    filename1 = "results/" + str(SimulationIndex) + "_MH_alpha_" + str(kappa)
    filename2 = "results/" + str(SimulationIndex) + "_MH_beta_" + str(kappa)
    filename4 = "results/" + str(SimulationIndex) + "_MH_lamb1_" + str(kappa)
    filename5 = "results/" + str(SimulationIndex) + "_MH_lamb2_" + str(kappa)
    my_save(alpha_list, filename1)
    my_save(beta_list, filename2)
    my_save(lamb1_list, filename4)
    my_save(lamb2_list, filename5)
    filename3 = "__" + str(SimulationIndex) + "_timelist_mh_" + str(kappa)
    plt.hist(alpha_list, color='blue', label='MH', alpha = 0.5, density=True)
    plt.legend(prop={'size': 20})
    plt.title("posterior alpha")
    plt.savefig('results/posterior_alpha.pdf')
    plt.close()
    return 

def main():
    seed, kappa, n_sample, n_burnin  = validateInput(sys.argv)
    pi_0 = [.5, .5]
    T_interval = [0.0, 2319.838]
    k = 2
    priors = np.array([[2, 2], [2, 3], [3, 2], [1, 2]])
    print("Experiment (Ecoli lag inner) starts!")
    print("seed =", seed)
    print("kappa =", kappa)
    print("sample size =", n_sample)
    print("first step sample size (for computing covariance matrix) =", n_burnin)
    experiment(seedIndex=seed, kappa=kappa, pi_0=pi_0, sample_n=n_sample, T_interval=T_interval, k=2, priors=priors, burnin_size=n_burnin)
    return
if __name__ == '__main__':
    main()
