from numpy import random
import copy
from model_parameters import *


class Observation:
    # [t_start, t_1, t_2,...,t_N, t_end]
    # [s_0, s_1, s_2,...,s_N]
    def __init__(self, O=None, T=None,t_start=0, t_end=10):
        self.O = O
        self.T = T
        self.t_start = t_start
        self.t_end = t_end

    def sample_observation(self, time_points, MJPpath0):
        self.t_start = MJPpath0.t_start
        self.t_end = MJPpath0.t_end
        self.T = time_points
        S = copy.deepcopy(MJPpath0.S)
        mjpT = copy.deepcopy(MJPpath0.T)
        mjpT.insert(0, self.t_start)
        index = [getindex(mjpT, x) for x in time_points]
        States = [S[x - 1] for x in index]
        observation = random.normal(States)
        self.O = list(observation)


class MJPpath:
    def __init__(self, S, T, t_start, t_end, rate_matrix, initial_pi):
        self.s0 = S[0]
        self.S = S
        self.T = T
        self.t_start = t_start
        self.t_end = t_end
        self.total_state = len(initial_pi)
        self.rate_matrix = rate_matrix
        self.initial_pi = initial_pi
        if len(S) - 1 != len(T):
            print("ERROR:The length of S and the length of T are not matching!:(")

    def generate_newpath(self, known_first=False):
        # generate a new path given the rate matrix.
        t_start = self.t_start
        t_end = self.t_end
        t = t_start
        z = 0
        rate_matrix = copy.deepcopy(self.rate_matrix)
        if not known_first:
            pi0 = self.initial_pi
            s0 = sample_from_Multi(pi0)
            self.s0 = s0
        rate_list = abs(rate_matrix.diagonal())
        S = []
        T = []
        temp_state = self.s0
        while (t + z) < t_end:
            t += z
            current_state = temp_state
            S.append(current_state)
            if t > t_start:
                T.append(t)
            rate = rate_list[current_state]
            z = random.exponential(1.0 / rate)
            beta = rate_matrix[current_state]
            beta[beta < 0] = 0
            temp_state = sample_from_Multi(beta)
        self.S = S
        self.T = T

    def delete_virtual(self):
        S = copy.deepcopy(self.S)
        S = S[1: len(S)]
        T = copy.deepcopy(self.T)
        s0 = self.s0
        t_0 = self.t_start
        newS = []
        newT = []
        old_state = s0
        for i in range(len(S)):
            if S[i] != old_state:
                newS.append(S[i])
                newT.append(T[i])
                old_state = S[i]
        newS.insert(0, s0)
        self.S = newS
        self.T = newT

    def stay_time(self, s):
        T = copy.deepcopy(self.T)
        T.insert(0, self.t_start)
        T.append(self.t_end)
        diff_T = [T[i + 1] - T[i] for i in range(len(T) - 1)]
        staying_time = 0
        for i in range(len(diff_T)):
            if self.S[i] == s:
                staying_time += diff_T[i]
        return staying_time

    def stay_time_list(self):
        time_list = []
        for s in range(self.total_state):
            time_list.append(self.stay_time(s))
        return time_list
