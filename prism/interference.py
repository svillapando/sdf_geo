
from __future__ import annotations
from typing import Callable, Union
import numpy as np
import csdl_alpha as csdl


SDF = Callable[[csdl.Variable], csdl.Variable]

_DEF_EPS = 1e-12


def _project_from_x(x: csdl.Variable, phi_x: csdl.Variable, grad_x: csdl.Variable,
                    eps: float = _DEF_EPS) -> csdl.Variable:
    """First-order projection of x onto the zero level set of an SDF.
    a = x - phi(x) * grad/||grad||^2
    """
    denom = csdl.sum(grad_x * grad_x) + eps
    return x - (phi_x * grad_x) / denom


def collision_check(
    phi_A: SDF,
    phi_B: SDF,
    x0: Union[np.ndarray, csdl.Variable],
    *,
    newton_tol: float = 1e-8,
    newton_name: str = "collision_check",
    return_all: bool = False,
) -> tuple[csdl.Variable, ...] | csdl.Variable:
    """
    Perform collision check between two SDFs.

    Parameters
    ----------
    phi_A, phi_B : callable
        Functions mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
    x0 : np.ndarray | csdl.Variable, shape (3,)
        Initial seed for the Newton solve. May be a NumPy array or a CSDL Variable.
    newton_tol : float
        Convergence tolerance.
    newton_name : str
        Name for the CSDL Newton solver instance.
    return_all : bool
        If True, return tuple (x_star, F_star, a, b, pair_gap).
        Else return just pair_gap.
    """
    if isinstance(x0, csdl.Variable):
        x = x0
        use_initial_value = False
        init_val = None
    else:
        x = csdl.Variable(name=f"{newton_name}:x", shape=(3,), value=np.asarray(x0, dtype=float))
        use_initial_value = True
        init_val = np.asarray(x0, dtype=float)

    phiA_x = phi_A(x)
    phiB_x = phi_B(x)
    F = csdl.maximum(phiA_x, phiB_x)

    gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))

    solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
    if use_initial_value:
        solver.add_state(x, gF, initial_value=init_val)
    else:
        solver.add_state(x, gF)
    solver.run()

    phiA_x = phi_A(x)
    phiB_x = phi_B(x)
    gA = csdl.reshape(csdl.derivative(ofs=phiA_x, wrts=x), (3,))
    gB = csdl.reshape(csdl.derivative(ofs=phiB_x, wrts=x), (3,))

    a = _project_from_x(x, phiA_x, gA)
    b = _project_from_x(x, phiB_x, gB)
    pair_gap = csdl.norm(a - b)

    if return_all:
        return x, F, a, b, pair_gap
    return pair_gap

__all__ = ["collision_check"]
