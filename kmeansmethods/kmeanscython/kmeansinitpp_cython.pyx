cimport cython
import numpy as np
import random

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def kmeansinitpp_cython(double [:, :] X, int k):
    cdef int l = X.shape[0]
    cdef int n = X.shape[1]
    cdef int i, j, h, g
    cdef long start = random.randint(0, l-1)
    cdef long sam
    cdef double[:, :] C = np.zeros((k, n))
    cdef double[:] dist, dislist, temp
    temp = np.zeros(n)
    for i in range(n):
        C[0, i] = X[start, i]
    for i in range(1, k):
        dislist = np.zeros(l)
        dist = np.zeros(i)
        for j in range(l):
            for h in range(i):
                for g in range(n):
                    temp[g] = C[h, g] - X[j, g]
                dist[h] = np.linalg.norm(temp)
            dislist[j] = np.square(np.min(dist))
        phi = np.sum(dislist)
        prob = dislist / phi
        sam = np.random.choice(l, 1, p=prob)   
        for j in range(n):
            C[i, j] = X[sam, j]
    return C
