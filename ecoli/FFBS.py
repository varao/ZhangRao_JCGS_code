from path_class import *
import numpy as np

def FF(likelihood, initial_pi, OMEGA, path): #
    # UPDATED NUMPY CALCULATION VERSION
    # X is the corresponding observations X1, ... ,XT
    NumofStates = len(initial_pi)
    rate_matrix = copy.deepcopy(path.rate_matrix)
    B = np.identity(NumofStates) + rate_matrix / float(OMEGA)
    NumofMessage = len(path.T) + 1
    alphaMat = np.zeros([NumofMessage, NumofStates])
    alphaMat[0] = initial_pi
    MaximumConsts = []
    MaximumConsts.append(1.)
    # forward filtering:
    # (len(path.T) + 1) * N dimensional likelihood matrix
    for t in range(1, len(path.T) + 1):
        alpha = np.matmul(alphaMat[t - 1] * likelihood[t - 1], B)
        max_alpha = np.max(alpha)
        MaximumConsts.append(max_alpha)
        alpha = alpha * 1. / max_alpha
        alphaMat[t] = alpha
    # backward sampling:
    newS = []
    alphaLast = np.array(likelihood[-1]) * alphaMat[-1]
    PMarginal = sum(alphaLast)
    logm = 0.0
    for m in MaximumConsts:
        logm += np.log(m)
    logProbMarginal = np.log(PMarginal) + logm
    return logProbMarginal, alphaMat

def BS(initial_pi, alpha, likelihood, OMEGA, path):
    # UPDATED NUMPY CALCULATION VERSION
    # Here path means path containing virtual jumps
    # X is the corresponding observations X1, ... ,XT
    NumofStates = len(initial_pi)
    rate_matrix = copy.deepcopy(path.rate_matrix)
    path_times = copy.deepcopy(path.T)
    B = np.identity(NumofStates) + rate_matrix / float(OMEGA)
    t_start = path.t_start
    t_end = path.t_end
    newS = []
    beta = np.array(likelihood[-1]) * np.array(alpha[-1])
    temp = sample_from_Multi(beta)
    newS.append(temp)
    iter = range(len(path_times))
    for t in iter.__reversed__():
        beta = alpha[t] * likelihood[t] * B[:, newS[-1]]
        newState = sample_from_Multi(beta)
        newS.append(newState)
    newS.reverse()
    MJPpathNew= MJPpath(newS, path_times, t_start, t_end, rate_matrix, initial_pi)
    return MJPpathNew


def FFBS(initial_pi, observation, OMEGA, path): # Here path means path containing virtual jumps
    # forward filtering:
    logProbMarginal, alphaMat = FF(likelihood, initial_pi, OMEGA, path)
    # backward sampling:
    MJPpathNew = BS(initial_pi, alphaMat, likelihood, OMEGA, path)
    return MJPpathNew # with virtual jumps
