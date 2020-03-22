import time
import sys
import string
from path_class import *
from gibbs_MJPs import *
from gbs_immi import *
from mh_immi import *
from mh_immi_old import *
from model_parameters import *
import json
import pickle
from numpy import random
def my_save(list, filename):
    import json
    f = open(filename, 'w')
    json.dump(list, f)
    f.close()

def parseArgs(args):
    #"""Parses arguments vector, looking for switches of the form -key {optional value}.
    #For example:
	#parseArgs([ 'template.py', '-v', '10', '-n' , '100'}"""
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
    if '-n' in args_map:
        number = int(args_map['-n'])
    return number

def simu_one(number):
    #set a seed
    random.seed(number)
    var = [0]
    time_gbs_list = []
    import json
    with open('_data_EXPTime_interval_immi_d3', 'r') as f:
        T_interval = json.load(f)
        f.close()
    with open('_data_EXPprior_immi_d3', 'r') as f:
        mu, lamb, omega, theta = json.load(f)
        f.close()
    with open('_data_observation_immi_d3', 'r') as f:
        observation = pickle.load(f)
        f.close()
    observation.info()
    pi_0 = [1. / 3] * 3
    sample_n = 5000
    k = 2
    for cur_var in var:
        print(number, "variance for current iteration is : ", cur_var)
        if len(time_gbs_list) > 0:
            print("gbs_time :", time_gbs_list)
        observation1 = copy.deepcopy(observation)
        t0 = time.clock()
        samples_GBS, gbsalpha_list, gbsbeta_list = BGsampler_immi(observation1, pi_0, sample_n, T_interval, k, mu, lamb, omega, theta)
        dt = time.clock() - t0
        time_gbs_list.append(dt)
        import json
        filename1 = "__" + str(number) + "_GBS_alpha" + str(cur_var)
        filename2 = "__" + str(number) + "_GBS_beta" + str(cur_var)
        my_save(gbsalpha_list, filename1)
        my_save(gbsbeta_list, filename2)
    filename1 = "__" + str(number) + "_timelist_gbs"
    my_save(time_gbs_list, filename1)


def main():
    arguments = validateInput(sys.argv)
    i = arguments
    simu_one(i)

if __name__ == '__main__':
    main()