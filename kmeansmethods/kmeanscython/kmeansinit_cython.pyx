cimport cython
import numpy as np
import random

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def kmeansinit_cython(double [:, :] X, int k):
    cdef int l = X.shape[0]
    cdef int m = X.shape[1]
    cdef long [:] pos = np.array(random.sample(list(np.arange(l)),k))
    cdef double[:, :] result = np.zeros((k, m))
    for i in range(k):
        for j in range(m):
            result[i, j] = X[pos[i], j]
    return result
