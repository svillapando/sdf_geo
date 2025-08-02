import numpy as np

def smooth_union(a, b, k=10.0):
    return -np.log(np.exp(-k * a) + np.exp(-k * b)) / k
 
