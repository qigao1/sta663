import numpy as np
import random
from kmeanspython.kmeansinitpp import kmeansinitpp
from kmeanspython.kmeansloop import kmeansloop

def kmeansinitvv(X, k, l):
    n = X.shape[0]
    C = X[random.randint(0, n-1)].reshape((1,-1))
    dislist = np.square(np.linalg.norm(X - C, axis = 1))
    phi = np.sum(dislist)
    for i in range(int(round(np.log(phi)))):
        prob = np.multiply(dislist / phi, l)
        ran = np.random.random(n)
        C = np.vstack((C, X[prob > ran]))
        dislist = []
        for j in range(len(X)):
            dist = np.apply_along_axis(np.linalg.norm, 1, C - X[j].reshape((1,-1)))
            dislist.append(np.square(np.min(dist)))
        phi = np.sum(dislist)
    init = kmeansinitpp(C, k)
    out = kmeansloop(C, init, 10)
    return out[0]
