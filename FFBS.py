from path_class import *
from math import log
import numpy as np
"""
Rao-Teh 2013

FFBS functions
Sample virtual jumps
MJP Likelihood
"""


def sampleUI(MJPpath0, OMEGA):
    # [t_0(t_start), t_1, t_2,...,t_N, t_N+1(t_end)]
    # [s_0, s_1, s_2,...,s_N]
    # Input  : Old trajectory and bound constant OMEGA
    # Output : Trajectory with true jumps and virtual jumps
    # for more convenience calling
    A = MJPpath0.rate_matrix
    S = MJPpath0.S
    T = copy.deepcopy(MJPpath0.T)
    T.append(MJPpath0.t_end)
    T.insert(0, MJPpath0.t_start)
    AS = A.diagonal()
    vj_times = [] # virtual jump times
    vj_states = [] # virtual jump states
    # Here is the trick to generate non-homogeneous poisson process
    for i in range(len(T) - 1 ):
        current_state = S[i]
        rate = OMEGA + AS[current_state]
        t = float(T[i])
        z = 0.0 # time_step
        while t + z < T[i + 1]:
            t += z
            # add 0 as virtual jump time. delete it later
            vj_times.append(t)
            # in the end we will get the union of virtual jumps and effective jumps
            vj_states.append(current_state)
            z = random.exponential(1.0 / rate)
    new_MJP = copy.deepcopy(MJPpath0)
    vj_times.pop(0)
    new_MJP.S = vj_states
    new_MJP.T = vj_times
    return new_MJP


def get_likelihood(observation, path_T, N, t_interval):
    t_start = t_interval[0]
    t_end = t_interval[1]
    path_T = copy.deepcopy(path_T)
    path_T.append(t_end)
    path_T.insert(0, t_start)
    O = observation.O
    OT = observation.T
    likelihood_list = []
    for i in range(len(path_T) - 1):
        likelihood = [1] * N
        for s in range(N):
            for j in range(len(O)):
                if path_T[i] <= OT[j] < path_T[i + 1]:
                    likelihood[s] *= trans_Likelihood(O[j], s)
        likelihood_list.append(likelihood)
    # 0 , 1, 2, ..., n-1
    return likelihood_list


def FF(likelihood, initial_pi, OMEGA, path): #
    # X is the corresponding observations X1, ... ,XT
    N = len(initial_pi)
    rate_matrix = copy.deepcopy(path.rate_matrix)
    B = np.identity(N) + rate_matrix / float(OMEGA)
    alpha = [initial_pi]
    M = [1]
    # forward filtering:
    # (len(path.T) + 1) * N dimensional likelihood matrix
    for t in range(1, len(path.T) + 1):
        temp = [0.0] * N
        # j-th coordinate
        for j in range(N):
            for k in range(max(j - 1, 0), min(j + 2, N)):
                temp[j] += alpha[t - 1][k] * likelihood[t - 1][k] * B[k][j]
        maxt = max(temp)
        M.append(maxt)
        temp2 = [x / maxt for x in temp]
        alpha.append(temp2)
    beta = np.array(likelihood[-1]) * np.array(alpha[-1])
    p_marginal = sum(beta)
    logm = 0.0
    for m in M:
        logm += log(m)
    log_p = log(p_marginal) + logm
    return log_p, alpha


def BS(initial_pi, alpha, likelihood, OMEGA, path):
    # Here path means path containing virtual jumps
    # X is the corresponding observations X1, ... ,XT
    N = len(initial_pi)
    rate_matrix = copy.deepcopy(path.rate_matrix)
    path_times = copy.deepcopy(path.T)
    B = np.identity(N) + rate_matrix / float(OMEGA)
    t_start = path.t_start
    t_end = path.t_end
    newS = []
    beta = np.array(likelihood[-1]) * np.array(alpha[-1])
    temp = sample_from_Multi(beta)
    newS.append(temp)
    iter = list(range(len(path_times)))
    iter.reverse()
    for t in iter:
        beta = [0.0] * N
        for i in range(N):
            beta[i] = alpha[t][i] * likelihood[t][i] * B[i][newS[-1]]
        temp = sample_from_Multi(beta)
        newS.append(temp)
    newS.reverse()
    MJPpath_new= MJPpath(newS, path_times, t_start, t_end, rate_matrix, initial_pi)
    return MJPpath_new


def RaoTeh_update(observation, MJPpath0, OMEGA):
    initial_pi = copy.deepcopy(MJPpath0.initial_pi)
    path = sampleUI(MJPpath0, OMEGA)
    likelihood = get_likelihood(observation, path.T, len(initial_pi), [path.t_start, path.t_end])
    log_p, alpha = FF(likelihood, initial_pi, OMEGA, path)
    new_path = BS(initial_pi, alpha, likelihood, OMEGA, path)
    new_path.delete_virtual()
    return new_path
