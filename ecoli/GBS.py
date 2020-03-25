from path_class import *
from FFBS import FF, BS
import numpy as np

#[t_0(t_start), t_1, t_2,...,t_N, t_N+1(t_end)]
# #[s_0, s_1, s_2,...,s_N]

# Input  : Old trajectory and bound constant OMEGA
# Output : Trajectory with true jumps and virtual jumps
def SampleVirtualJumps(path, OMEGA):
    # for more convenience calling
    A = path.rate_matrix
    S = path.S
    T = copy.deepcopy(path.T)
    T.append(path.t_end)
    T.insert(0, path.t_start)
    Adiag = A.diagonal()
    VJTimes = [] # virtual jump times
    VJStates = [] # virtual jump states
    #Here is the trick to generate non-homogeneous poisson process
    for i in range(len(S)):
        CurrentState = S[i]
        rate = OMEGA + Adiag[CurrentState]
        CurrentT = float(T[i])
        z = 0.0 # time_step
        while CurrentT + z < T[i + 1]:
            CurrentT += z
            # add 0 as virtual jump time. delete it later
            VJTimes.append(CurrentT) # in the end we will get the union of virtual jumps and effective jumps
            VJStates.append(CurrentState)
            scale = 1.0 / rate
            z = np.random.exponential(scale)
    NewPath = copy.deepcopy(path)
    VJTimes.pop(0)
    NewPath.S = VJStates
    NewPath.T = VJTimes
    return NewPath


# model dependent
def CalLikelihood(observation, PathT, t_interval, lambs):
    grid = copy.deepcopy(PathT)
    grid.append(t_interval[1])
    grid.insert(0, t_interval[0])
    ObsTime = observation.T
    LikelihoodList = []
    NumofObs = len(ObsTime)
    CurItvIndex = 0
    for i in range(len(grid) - 1):
        likelihood = np.ones(2)
        obscount = 0
        deltaT = grid[i + 1] - grid[i]
        while CurItvIndex < NumofObs and ObsTime[CurItvIndex] < grid[i + 1]:
            obscount += 1
            CurItvIndex += 1
        likelihood *= np.array([(lambs[0] ** obscount) * np.exp(-lambs[0] * deltaT), (lambs[1] ** obscount) * np.exp(-lambs[1] * deltaT)])
        LikelihoodList.append(likelihood)
    return LikelihoodList

# given rate matrix(i.e. all the parameters), and the trajectory with virtual jumps,
# sample a new trajectory.
def UpdatePath(observation, path, OMEGA, lambs):
    initial_pi = copy.deepcopy(path.initial_pi)
    PathwtVirtualJumps = SampleVirtualJumps(path, OMEGA)
    likelihood = CalLikelihood(observation, PathwtVirtualJumps.T, [PathwtVirtualJumps.t_start, PathwtVirtualJumps.t_end], lambs)
    # Forward calculate P(Y | W)
    p,  ALPHA = FF(likelihood, initial_pi, OMEGA, PathwtVirtualJumps)
    NewPathwotVirtualJumps = BS(initial_pi, ALPHA, likelihood, OMEGA, PathwtVirtualJumps)
    NewPathwotVirtualJumps.delete_virtual()
    return NewPathwotVirtualJumps

def UpdatePathandPars(observation, path, k, lambs, alpha, beta, priors):
    #initialization
    a1, b1 = priors[0]
    a2, b2 = priors[1]
    c1, d1 = priors[2]
    c2, d2 = priors[3]
    d = len(path.rate_matrix)
    OMEGA = get_omega(path.rate_matrix, k)
    NewPath = UpdatePath(observation, path, OMEGA, lambs)
    # Sample alpha & beta| S, T, y:
    # Apply Metropolis Hasting to sample beta | S, T, y
    n = len(NewPath.T)
    # Gibbs update all parameters
    grid = copy.deepcopy(NewPath.T)
    grid.insert(0, NewPath.t_start)
    grid.append(NewPath.t_end)

    CountInc = 0
    CountDec = 0

    Tal1 = 0.0
    Tal2 = 0.0

    for i in range(1, len(NewPath.S)):
        if NewPath.S[i] > NewPath.S[i - 1]:
            CountInc += 1
        else:
            CountDec += 1
    for i in range(len(NewPath.S)):
        deltaT = grid[i + 1] - grid[i]
        if NewPath.S[i] == 0:
            Tal1 += deltaT
        else:
            Tal2 += deltaT
    ## Calculate n1, n2
    n1 = 0
    n2 = 0
    ObsTime = observation.T
    NumofObs = len(ObsTime)
    CurItvIndex = 0
    for i in range(len(grid) - 1):
        while CurItvIndex < NumofObs and ObsTime[CurItvIndex] < grid[i + 1]:
            if NewPath.S[i] == 0:
                n1 += 1
            else:
                n2 += 1
            CurItvIndex += 1

    alpha = np.random.gamma((a1 + CountInc), 1./ (b1 + Tal1))
    beta = np.random.gamma((a2 + CountDec), 1./ (b2 + Tal2))
    lambs[0] = np.random.gamma((c1 + n1), 1./ (d1 + Tal1))
    lambs[1] = np.random.gamma((c2 + n2), 1./ (d2 + Tal2))
    NewPath.rate_matrix = constructor_rate_matrix(alpha, beta)
    return NewPath, alpha, beta, lambs

def GBSsampler(observation, pi_0, sample_n, T_interval, k, priors):
    # Initialization:
    alpha = 0.05
    beta = 0.71
    lambs = np.array([0.027, 0.495])
    rate_matrix = constructor_rate_matrix(alpha, beta)
    sample_old = MJPpath(S=[0], T=[], t_start=T_interval[0], t_end=T_interval[1], rate_matrix=rate_matrix, initial_pi=pi_0)
    sample_old.generate_newpath()
    sample_list = []
    sample_list.append(copy.deepcopy(sample_old))
    alpha_list = []
    beta_list = []
    lamb1_list = []
    lamb2_list = []
    # Gibbs Iteration:
    for i in range(sample_n):
        if i == sample_n / 2:
            print("GBS 50%")
        if i == sample_n - 1:
            print("GBS 100%")
        sample_new, alpha, beta, lambs = UpdatePathandPars(observation, sample_old, k, lambs, alpha, beta, priors)
        sample_list.append(copy.deepcopy(sample_new))
        sample_old = sample_new
        alpha_list.append(alpha)
        beta_list.append(beta)
        lamb1_list.append(lambs[0])
        lamb2_list.append(lambs[1])
    return alpha_list, beta_list, lamb1_list, lamb2_list
