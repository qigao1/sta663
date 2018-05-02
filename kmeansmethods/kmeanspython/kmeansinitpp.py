import numpy as np
import random

def kmeansinitpp(X, k):
    """
    k-means++
    notations: X is the data set; k is the number of desired clusters
    """
    n = X.shape[0]
    C = X[random.randint(0, n-1)].reshape((1,-1))
    while len(C) < k:
        dislist = []
        for i in range(len(X)):
            dist = np.apply_along_axis(np.linalg.norm, 1, C - X[i].reshape((1,-1)))
            dislist.append(np.square(np.min(dist)))
        phi = np.sum(dislist)
        prob = dislist / phi
        C = np.vstack((C, X[np.random.choice(n, 1, p=prob)]))
    return C
