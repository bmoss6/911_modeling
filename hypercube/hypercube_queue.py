import numpy as np

class HyperCubeQueue():
    """The HyberCube Queuing Model as described in Larson(1974).

    Args:
        T(numpy.ndarray): Matrix of travel times where t_ij is mean travel time from atom i to atom j
        L(numpy.ndarray): Location matrix where l_nj is probability that response unit n is located in atom j while available or idle
        f(numpy.ndarray): Vector of workload fractions for each geographical atom
        lam(float): Poisson variable for arrival rate of calls for service
        mu(float): Exponential variable for average service rate of calls
    """
    def __init__(self, T=None, L=None, a=None, f=None, lam=None, mu=None):
        self.T = T
        self.L = L
        self.a = a
        self.f = f
        self.lam = lam
        self.mu = mu
        
    def tour(self):
        pass
    def weight(self, b1):
        # Taken from https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
        count = 0
        while (b1):
            b1 &= (b1-1)
            count += 1
        return count

    def hamming_distance(self, s1,s2):
        upward = self.weight(s2 & s1)
        downward = self.weight(s1 & s2)
        return upward, downward

    def create_state_space(self, N=2):
        S = [0, 0,1]
        m2 = 2
        n = 2
        while n <= N:
            m1 = m2
            m2 = (2* m1)
            i = m1
            S.append(m1 + S[m2-i])
            i = i+1
            while i<m2:
                S.append(m1 + S[m2-i])
                i = i+1
            n += 1
        return S[1:]

if __name__ =="__main__":
    h = HyperCubeQueue()
    t = h.create_state_space()
    for x in range(len(t)):
        print(bin(t[x]), bin(t[x+1]))
        print(h.hamming_distance(t[x], t[x+1]))
