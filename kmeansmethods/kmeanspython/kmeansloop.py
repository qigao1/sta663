import numpy as np
from operator import itemgetter

def kmeansloop(X, init, maxiter = 100, tol = 1e-4):
    '''
    finding the best centers
    notations: newcenter is the updated center each loop; clulist is the cluster that each observation belongs to;
    count is total loop to get a tolerance smaller than 1e-4; cost is the total Euclidean distance
    '''
    oldcenter = init
    l, n = X.shape
    k = len(init)
    count = 0
    while count < maxiter:
        count += 1
        clulist = np.zeros(l)
        dislist = np.zeros(l)
        for i in range(l):
            dist = np.apply_along_axis(np.linalg.norm, 1, oldcenter - X[i].reshape((1,-1)))
            clulist[i], dislist[i] = min(enumerate(dist), key=itemgetter(1)) 
        cluscenter = np.unique(clulist)
        newcenter = np.where(np.repeat(np.array([i in cluscenter for i in range(k)]),n).reshape((k,n)), 
                             np.array(list(map(lambda z: np.where(np.sum(clulist == z) > 0, 
                                                                  np.mean(X[clulist == z],axis = 0), 
                                                                  np.zeros(n)), range(k)))),
                             oldcenter)
        if np.allclose(oldcenter, newcenter, rtol=0, atol=tol):
            break
        oldcenter = newcenter
    cost = np.sum(np.square(dislist))
    return newcenter, clulist, count, cost
