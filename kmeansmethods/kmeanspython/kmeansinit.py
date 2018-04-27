import random as random
import numpy as np

def kmeansinit(X, k):
    l = len(X)
    lis = random.sample(list(np.arange(l)),k)
    return X[lis]
