
from __future__ import annotations
from typing import Callable, Dict
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
    x0: np.ndarray,
    *,
    newton_tol: float = 1e-8,
    newton_name: str = "collision_check",
    return_all: bool = False,
) -> Dict[str, csdl.Variable] | csdl.Variable:
    """Perform collision check between two SDFs.

    Parameters
    ----------
    phi_A, phi_B : callable
        Functions mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
    x0 : np.ndarray, shape (3,)
        Initial seed for the Newton solve.
    newton_tol : float
        Convergence tolerance.
    newton_name : str
        Name for the CSDL Newton solver instance.
    return_all : bool
        If True, return dict of all relevant variables.

    Returns
    -------
    dict or csdl.Variable
        If return_all=True, returns dict with keys: 'x_star', 'F_star', 'a', 'b', 'pair_gap'.
        Else returns just 'pair_gap'.
    """
    # State variable for the Newton solve
    x = csdl.Variable(name=f"{newton_name}:x", shape=(3,), value=np.asarray(x0, dtype=float))

    # Build F(x)
    phiA_x = phi_A(x)
    phiB_x = phi_B(x)
    F = csdl.maximum(phiA_x, phiB_x)

    # Residual is the gradient of F w.r.t x
    gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))

    solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
    solver.add_state(x, gF, initial_value=np.asarray(x0, dtype=float))
    solver.run()

    # Re-evaluate at converged x
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


__all__ = [
    "collision_check",
]
