import numpy as np
import csdl_alpha as csdl

# Both A and B
def smooth_union(a, b):
    return csdl.minimum(a,b)
 
# Only overlapping regions of A and B
def smooth_intersection(a, b):
    return csdl.maximum(a,b)

# Parts of A that are NOT inside B (A - B)
def smooth_subtraction(a, b):
    return smooth_intersection(a, -b)

