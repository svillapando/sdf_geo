
import csdl_alpha as csdl
from typing import Callable

SDF = Callable[[csdl.Variable], csdl.Variable]

def _combine_min(f: SDF, g: SDF) -> SDF:
    return lambda p: csdl.minimum(f(p), g(p))

def _combine_max(f: SDF, g: SDF) -> SDF:
    return lambda p: csdl.maximum(f(p), g(p))

def union(f1: SDF, f2: SDF, *rest: SDF) -> SDF:
    
    out = _combine_min(f1, f2)
    for f in rest:
        out = _combine_min(out, f)
    return out

def intersection(f1: SDF, f2: SDF, *rest: SDF) -> SDF:
    
    out = _combine_max(f1, f2)
    for f in rest:
        out = _combine_max(out, f)
    return out

def subtraction(f: SDF, g: SDF, *more: SDF) -> SDF:
    # A - (B U C U D ...)
    cutters = union(g, *more) if more else g
    return lambda p: csdl.maximum(f(p), -cutters(p))


# ============ Numpy Versions for Explicit Op =============#
# --- NumPy SDF compositors (drop alongside the CSDL ones) ---
import numpy as np

SDF_NP = Callable[[np.ndarray], np.ndarray]

def _combine_min_np(f: SDF_NP, g: SDF_NP) -> SDF_NP:
    return lambda p: np.minimum(f(p), g(p))

def _combine_max_np(f: SDF_NP, g: SDF_NP) -> SDF_NP:
    return lambda p: np.maximum(f(p), g(p))

def union_np(f1: SDF_NP, f2: SDF_NP, *rest: SDF_NP) -> SDF_NP:
    """
    NumPy union: pointwise min over SDFs (same semantics as CSDL union).
    """
    out = _combine_min_np(f1, f2)
    for f in rest:
        out = _combine_min_np(out, f)
    return out

def intersection_np(f1: SDF_NP, f2: SDF_NP, *rest: SDF_NP) -> SDF_NP:
    """
    NumPy intersection: pointwise max (same semantics as CSDL intersection).
    """
    out = _combine_max_np(f1, f2)
    for f in rest:
        out = _combine_max_np(out, f)
    return out

def subtraction_np(f: SDF_NP, g: SDF_NP, *more: SDF_NP) -> SDF_NP:
    """
    NumPy subtraction: A - (B ∪ C ∪ ...).
    """
    cutters = union_np(g, *more) if more else g
    return lambda p: np.maximum(f(p), -cutters(p))
