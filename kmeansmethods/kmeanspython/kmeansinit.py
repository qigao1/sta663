import random as random
import numpy as np

def kmeansinit(X, k):
    """
    k-means
    notations: X is the data set; k is the desired number of clusters
    """
    l = len(X)
    lis = random.sample(list(np.arange(l)),k)
    return X[lis]
