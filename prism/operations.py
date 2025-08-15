
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
