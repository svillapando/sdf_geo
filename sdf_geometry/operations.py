import numpy as np

# Both A and B
def smooth_union(a, b, k=10.0):
    return -np.log(np.exp(-k * a) + np.exp(-k * b)) / k
 
# Only overlapping regions of A and B
def smooth_intersection(a, b, k=10.0):
    return np.log(np.exp(k * a) + np.exp(k * b)) / k

# Parts of A that are NOT inside B (A - B)
def smooth_subtraction(a, b, k=10.0):
    return smooth_intersection(a, -b, k)

