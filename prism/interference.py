
from __future__ import annotations
from typing import Callable, Union, Tuple
import numpy as np
import csdl_alpha as csdl


SDF = Callable[[csdl.Variable], csdl.Variable]

_DEF_EPS = 1e-12


def collision_check(
    phi_A: SDF,
    phi_B: SDF,
    x0: Union[np.ndarray, csdl.Variable],
    *,
    gamma: float = 5.0,                 # penalty strength (try 1–20 depending on units)
    normalize_grads: bool = True,       # use unit normals to improve conditioning
    newton_tol: float = 1e-8,
    newton_name: str = "collision_check_penalty",
    return_all: bool = False,
) -> Tuple[csdl.Variable, ...] | csdl.Variable:
    """
    Fully differentiable collision check via the penalized closest-pair stationarity:
        (a - b) + gamma * phi_A(a) * gA_hat = 0
        (b - a) + gamma * phi_B(b) * gB_hat = 0
    Returns pair_gap by default; with return_all=True returns (m, F_star, a, b, pair_gap).
    """

    # ---- numeric seeds (leaf states) ----
    m0 = x0.value.reshape(3,) if isinstance(x0, csdl.Variable) else np.asarray(x0, float).reshape(3,)
    a0_np = m0.copy()
    b0_np = m0.copy()

    a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
    b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

    # φ and gradients
    phiA = phi_A(a)  # scalar
    phiB = phi_B(b)  # scalar

    gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
    gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))

    if normalize_grads:
        gA = gA / csdl.sqrt(csdl.vdot(gA, gA) + _DEF_EPS)
        gB = gB / csdl.sqrt(csdl.vdot(gB, gB) + _DEF_EPS)

    # Stationarity residuals (6 equations,3 each)
    R_a = (a - b) + gamma * phiA * gA   # (3,)
    R_b = (b - a) + gamma * phiB * gB   # (3,)

    # Newton solve 
    solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
    solver.add_state(a, R_a, initial_value=a0_np)
    solver.add_state(b, R_b, initial_value=b0_np)
    solver.run()

    # Outputs
    diff = a - b
    pair_gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

    m = 0.5 * (a + b)
    # keep your scoring-compatible value if you use it downstream
    F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.0507

    if return_all:
        return m, F_star, a, b, pair_gap
    return F_star

# Leftover helper for below
# def _project_from_x(x: csdl.Variable, phi_x: csdl.Variable, grad_x: csdl.Variable,
#                     eps: float = _DEF_EPS) -> csdl.Variable:
#     """First-order projection of x onto the zero level set of an SDF.
#     a = x - phi(x) * grad/||grad||^2
#     """
#     denom = csdl.sum(grad_x * grad_x) + eps
#     return x - (phi_x * grad_x) / denom


# Gradient of F formula (issues with deep collisions)
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check",
#     return_all: bool = False,
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Perform collision check between two SDFs.

#     Parameters
#     ----------
#     phi_A, phi_B : callable
#         Functions mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
#     x0 : np.ndarray | csdl.Variable, shape (3,)
#         Initial seed for the Newton solve. May be a NumPy array or a CSDL Variable.
#     newton_tol : float
#         Convergence tolerance.
#     newton_name : str
#         Name for the CSDL Newton solver instance.
#     return_all : bool
#         If True, return tuple (x_star, F_star, a, b, pair_gap).
#         Else return just pair_gap.
#     """
#     if isinstance(x0, csdl.Variable):
#         x = x0
#         use_initial_value = False
#         init_val = None
#     else:
#         x = csdl.Variable(name=f"{newton_name}:x", shape=(3,), value=np.asarray(x0, dtype=float))
#         use_initial_value = True
#         init_val = np.asarray(x0, dtype=float)

#     phiA_x = phi_A(x)
#     phiB_x = phi_B(x)
#     F = csdl.maximum(phiA_x, phiB_x)

#     gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))

#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     if use_initial_value:
#         solver.add_state(x, gF, initial_value=init_val)
#     else:
#         solver.add_state(x, gF)
#     solver.run()

#     phiA_x = phi_A(x)
#     phiB_x = phi_B(x)
#     gA = csdl.reshape(csdl.derivative(ofs=phiA_x, wrts=x), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB_x, wrts=x), (3,))

#     a = _project_from_x(x, phiA_x, gA)
#     b = _project_from_x(x, phiB_x, gB)
#     pair_gap = csdl.norm(a - b)

#     if return_all:
#         return x, F, a, b, pair_gap
#     return pair_gap

# __all__ = ["collision_check"]


# KKT approach (issues with SDFs far apart and tangent)
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_kkt",
#     return_all: bool = False,
#     normalize_grads: bool = False,
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Closest-pair collision check via KKT system:

#         a - x - λ_A ∇φ_A(a) = 0
#         b - x - λ_B ∇φ_B(b) = 0
#         φ_A(a) = 0
#         φ_B(b) = 0
#         x - (a + b)/2 = 0

#     Parameters
#     ----------
#     phi_A, phi_B : callable
#         SDFs mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
#     x0 : np.ndarray | csdl.Variable
#         Numeric seed for the midpoint state x (if csdl.Variable, its .value is used).
#     newton_tol : float
#         Newton tolerance.
#     newton_name : str
#         Name for the Newton solver instance.
#     return_all : bool
#         If True, returns (x_star, F_star, a_star, b_star, pair_gap).
#         Else returns just pair_gap.
#     normalize_grads : bool
#         If True, uses unit normals g/||g|| in the KKT equations.

#     Returns
#     -------
#     tuple or csdl.Variable
#         If return_all=True: (x, F_star, a, b, pair_gap).
#         Else: pair_gap.
#     """
#     # ---- numeric seeds (keep states as leaf variables) ----
#     if isinstance(x0, csdl.Variable):
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     else:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np   = x0_np.copy()
#     b0_np   = x0_np.copy()
#     lamA0_np = np.array([0.0], dtype=float)
#     lamB0_np = np.array([0.0], dtype=float)

#     # ---- states ----
#     a    = csdl.Variable(name=f"{newton_name}:a",    shape=(3,), value=a0_np)
#     lamA = csdl.Variable(name=f"{newton_name}:lamA", shape=(1,), value=lamA0_np)
#     b    = csdl.Variable(name=f"{newton_name}:b",    shape=(3,), value=b0_np)
#     lamB = csdl.Variable(name=f"{newton_name}:lamB", shape=(1,), value=lamB0_np)
#     x    = csdl.Variable(name=f"{newton_name}:x",    shape=(3,), value=x0_np)

#     # ---- values & gradients on each surface point ----
#     phiA_a = phi_A(a)                              # scalar
#     phiB_b = phi_B(b)                              # scalar
#     gA = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))

#     if normalize_grads:
#         gA = gA / csdl.sqrt(csdl.vdot(gA, gA) + _DEF_EPS)
#         gB = gB / csdl.sqrt(csdl.vdot(gB, gB) + _DEF_EPS)

#     # broadcast λ to 3-vector
#     ones3 = csdl.Variable(value=np.ones(3))
#     lamA3 = csdl.reshape(csdl.tensordot(lamA, ones3, axes=None), (3,))
#     lamB3 = csdl.reshape(csdl.tensordot(lamB, ones3, axes=None), (3,))

#     # ---- residuals ----
#     RA1 = a - x - lamA3 * gA              # (3,)
#     RA2 = csdl.reshape(phiA_a, (1,))      # (1,)
#     RB1 = b - x - lamB3 * gB              # (3,)
#     RB2 = csdl.reshape(phiB_b, (1,))      # (1,)
#     RX  = x - 0.5*(a + b)                 # (3,)

#     # ---- Newton solve ----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a,    RA1, initial_value=a0_np)
#     solver.add_state(lamA, RA2, initial_value=lamA0_np)
#     solver.add_state(b,    RB1, initial_value=b0_np)
#     solver.add_state(lamB, RB2, initial_value=lamB0_np)
#     solver.add_state(x,    RX,  initial_value=x0_np)
#     solver.run()

#     # ---- outputs ----
#     # pair gap (guarded)
#     diff = a - b
#     gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#     # F_star for compatibility with your previous return tuple
#     F_star = csdl.maximum(phi_A(x), phi_B(x), rho=20.0)

#     if return_all:
#         return x, F_star, a, b, gap
#     return gap


# __all__ = [
#     "collision_check_kkt",
# ]


