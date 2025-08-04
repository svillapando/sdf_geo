import numpy as np
import csdl_alpha as csdl
from typing import Callable

SDF = Callable[[np.ndarray], np.ndarray]  # SDF: (..., 3) -> (...)

def smooth_union(f1: SDF, f2: SDF) -> SDF:
    def _union(p):
        return csdl.minimum(f1(p), f2(p))
    return _union

def smooth_intersection(f1: SDF, f2: SDF) -> SDF:
    def _intersection(p):
        return csdl.maximum(f1(p), f2(p))
    return _intersection

def smooth_subtraction(f1: SDF, f2: SDF) -> SDF:
    def _subtract(p):
        return csdl.maximum(f1(p), -f2(p))
    return _subtract
