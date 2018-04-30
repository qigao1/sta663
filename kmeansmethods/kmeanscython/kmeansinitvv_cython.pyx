cimport cython
import numpy as np
import random
from collections import Counter
from kmeanscython.kmeansinitpp_cython import kmeansinitpp_cython
from kmeanscython.kmeansloop_cython import kmeansloop_cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def kmeansinitvv_cython(double [:, :] X, int k, double l):
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]
    cdef int i, j, h, g
    cdef double[:, :] C
    cdef double[:] dist, dislist, temp, ran
    cdef double phi, psi
    cdef long [:] sampid = np.zeros(1,dtype=np.int)
    sampid[0] = np.random.randint(0, m-1)
    temp = np.zeros(n)
    dislist = np.zeros(m)
    for j in range(m):
        for g in range(n):
            temp[g] = X[sampid[0], g] - X[j, g]
        dislist[j] = np.square(np.linalg.norm(temp))
    phi = np.sum(dislist)
    psi = np.sum(dislist)
    for i in range(int(round(np.log(psi)))):
        ran = np.random.random(m)
        for j in range(m):
            if dislist[j] * l / phi > ran[j]:
                sampid = np.append(sampid, j)
        for j in range(m):
            dist = np.zeros(len(sampid))
            for h in range(len(sampid)):
                for g in range(n):
                    temp[g] = X[sampid[h], g] - X[j, g]
                dist[h] = np.linalg.norm(temp)
            dislist[j] = np.square(np.min(dist))
        phi = np.sum(dislist)
    C = np.zeros((len(sampid), n))
    for i in range(len(sampid)):
        for j in range(n):
            C[i, j] = X[sampid[i], j]
    clulist = np.zeros(m)
    for i in range(m):
        for j in range(len(sampid)):
            for h in range(n):
                temp[h] = C[j, h] - X[i, h]
            dist[j] = np.linalg.norm(temp)
        clulist[i], dislist[i] = min(enumerate(dist), key=itemgetter(1))
    weight = np.array(list(dict(Counter(clulist)).values()), dtype = np.double)
    init = kmeansinitpp_cython(C, k)
    out = kmeansloop_cython(np.asarray(C), np.asarray(init), 10, weight = weight)
    outt = out[0]
    return outt
