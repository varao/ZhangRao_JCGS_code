from math import pi
import numpy as np


def getindex(time_list, t):
    # [t_start, t_1, t_2,...]
    time_list = np.array(time_list)
    index = sum(time_list <= t)
    return index


def sample_from_Multi(beta, n=1):
    b_ = [i * 1.0 / sum(beta) for i in beta]
    if n == 1:
        return np.random.choice(len(b_), 1, p=b_)[0]
    return np.random.choice(len(b_), n, p=b_)


def trans_Likelihood(o, s):
    l = np.exp(- (s - o)**2 / 2.) / np.sqrt(2 * pi)
    return l


def get_omega(matrix, k):
    OMEGA = k * (max(-matrix[-1][-1], -matrix[-2][-2]))
    return OMEGA


def constructor_rate_matrix(alpha, beta, d=3):
    mat = np.identity(d, 'float')
    for i in range(1, d - 1):
        mat[i][i + 1] = alpha
        mat[i][i - 1] = i * beta
        mat[i][i] = -alpha - i * beta
    mat[0][1] = alpha
    mat[0][0] = -alpha
    mat[-1][-2] = (d - 1) * beta
    mat[-1][-1] = -(d - 1) * beta
    return mat


def propose(alpha_old, beta_old, var=0.01):
    alpha_new = np.exp(np.random.normal(np.log(alpha_old), np.sqrt(var)))
    beta_new = np.exp(np.random.normal(np.log(beta_old), np.sqrt(var)))
    return alpha_new, beta_new

