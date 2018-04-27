cimport cython
import numpy as np
import random
from operator import itemgetter

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def kmeansloop_cython(double [:, :] X, double [:, :] init, int maxiter = 100, double tol = 1e-4):
    cdef long l = X.shape[0]
    cdef long n = X.shape[1]
    cdef long k = init.shape[0]
    cdef double [:, :] oldcenter
    cdef double [:, :] newcenter
    oldcenter = np.zeros((k,n))
    newcenter = np.zeros((k,n))
    cdef long count
    count = 0
    cdef long [:] clulist
    cdef double [:] dislist
    cdef double [:] dist
    dist = np.zeros(k)
    cdef double [:] temp = np.zeros(n)
    cdef int i, j, h, a
    cdef double num
    for i in range(k):
        for j in range(n):
            oldcenter[i, j] = init[i, j]
    while count < maxiter:
        count += 1
        clulist = np.zeros(l, dtype=np.int)
        dislist = np.zeros(l)
        for i in range(l):
            for j in range(k):
                for h in range(n):
                    temp[h] = oldcenter[j, h] - X[i, h]
                dist[j] = np.linalg.norm(temp)
            newite = min(enumerate(dist), key=itemgetter(1))
            clulist[i] = newite[0]
            dislist[i] = newite[1]
        cluscenter = np.unique(clulist)
        for i in range(k):
            for j in range(n):
                newcenter[i, j] = oldcenter[i, j]
        for i in range(k):
            if i in cluscenter:
                for j in range(n):
                    num = 0
                    a = 0
                    for h in range(l):
                        if clulist[h] == i:
                            num += X[h, j]
                            a += 1
                    newcenter[i, j] = num / a
        if np.allclose(oldcenter, newcenter, rtol=0, atol=tol):
            break
        for i in range(k):
            for j in range(n):
                oldcenter[i, j] = newcenter[i, j]
    cost = np.sum(np.square(dislist))
    return newcenter, clulist, count, cost
