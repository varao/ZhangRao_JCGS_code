import numpy as np
import numpy.random as rd
# General Configures:
# --------------------------

def my_print(i, N=1000):
    if i % N == 0:
        print(i)

#[t_start, t_1, t_2,...]
def getindex(Tlist, t):
    Tlist = np.array(Tlist)
    index = np.sum(Tlist <= t)
    return index

def sample_from_Multi(beta):
    beta = np.array(beta)
    sum_beta = np.sum(beta)
    betaProb = beta * 1. / sum_beta
    ret = rd.choice(len(betaProb), 1, p=betaProb)[0]
    return ret


def get_omega(mat, k):
    return(max(-mat.diagonal()) * k)

# Model Specific Configures:
# --------------------------
def trans_Likelihood(o,s):
    l = np.exp(- (s - o)**2 / 2.) / np.sqrt(2 * pi)
    return l


def constructor_rate_matrix(alpha, beta):
    mat = np.identity(2, 'float')
    mat[0][0] = -alpha
    mat[0][1] = alpha
    mat[1][0] = beta
    mat[1][1] = -beta
    return mat

def propose(x, cov):
    """
    absolute normal proposal
    """
    temp = rd.multivariate_normal(x, cov)
    x_new = np.abs(np.array(temp))
#    x_new = np.exp(np.random.normal(np.log(x), np.sqrt(var)))
    return x_new


def log_propose_p(x, x_new):
    ll = - x_new * 1.0 / x - np.log(x)
    return ll
