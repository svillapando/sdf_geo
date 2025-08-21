
from __future__ import annotations
from typing import Callable, Union, Tuple
import numpy as np
import csdl_alpha as csdl


SDF = Callable[[csdl.Variable], csdl.Variable]

_DEF_EPS = 1e-12

# Collision check similar to paper
def collision_check(
    phi_A: SDF,
    phi_B: SDF,
    x0: Union[np.ndarray, csdl.Variable],
    eta_max: csdl.Variable,
    *,
    # smooth joint field (csdl.maximum implements log-sum-exp with "rho")
    rho: float = 20.0,             # higher = sharper max; start 5–20, homotope up/down as needed
    # tiny, fixed number of gradient pulls toward tightest gap/deepest overlap

    pull_steps: Tuple[float, ...] = (0.2, 0.1),  # fractions of length_scale
    length_scale: float = 1.0,     # sets absolute step sizes; e.g., gate radius or drone size
    return_all: bool = False,
    # kept for API compatibility with your KKT version (ignored here)
    newton_tol: float = 1e-8,
    newton_name: str = "collision_check_smooth",
    normalize_grads: bool = False,
) -> Tuple[csdl.Variable, ...] | csdl.Variable:
    """
    Differentiable SDF-SDF proximity/collision check

    Steps:
      1) Define smooth joint field F(x) = max(phi_A(x), phi_B(x)) 
      2) Do K fixed 'pull' steps: x <- x - eta * ∇F / ||∇F||.
      3) Project once from x_K to each surface:
           a = x_K - phi_A(x_K) / ||∇phi_A(x_K)||^2 * ∇phi_A(x_K)
           b = x_K - phi_B(x_K) / ||∇phi_B(x_K)||^2 * ∇phi_B(x_K)
      4) Outputs: gap = ||a - b||  (always >= 0), F_star = F(x_K),
         and optionally x_K, a, b.

    Returns:
      If return_all: (x_K, F_star, a, b, gap), else just F.
    """
    # ---- seed x0 as leaf variable ----
    if isinstance(x0, csdl.Variable):
        x0_np = np.asarray(x0.value, float).reshape(3,)
    else:
        x0_np = np.asarray(x0, float).reshape(3,)
    x = csdl.Variable(name=f"{newton_name}:x0", shape=(3,), value=x0_np)

    # ---- finite gradient steps on F(x) = max(phi_A, phi_B) ----

    for c_i in (0.9, 0.5, 0.25): 
    

      # inside each pull iteration
        FA = phi_A(x)
        FB = phi_B(x)
        F  = csdl.maximum(FA, FB, rho=rho)

        gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))
        gF_norm = csdl.sqrt(csdl.vdot(gF, gF)) + _DEF_EPS

        # adaptive step length: eta = min(c * |F|, eta_max)
        eta = csdl.minimum(c_i * csdl.absolute(F), eta_max)   # choose c_i per pull, e.g. 0.7, 0.35, 0.2

        x = x - (eta * gF / gF_norm)

    xK = x  # witness after pulls
    F_star = csdl.maximum(phi_A(xK), phi_B(xK), rho=rho) - 0.07


    # ---- two-step projection to each surface from xK ----
    # A
    FA = phi_A(xK)
    gA = csdl.reshape(csdl.derivative(ofs=FA, wrts=xK), (3,))
    gA_n = csdl.norm(gA) + _DEF_EPS
    a1 = xK - (FA / (gA_n * gA_n)) * gA

    FA = phi_A(a1)
    gA = csdl.reshape(csdl.derivative(ofs=FA, wrts=a1), (3,))
    gA_n = csdl.norm(gA) + _DEF_EPS
    a = a1 - (FA / (gA_n * gA_n)) * gA

    # B
    FB = phi_B(xK)
    gB = csdl.reshape(csdl.derivative(ofs=FB, wrts=xK), (3,))
    gB_n = csdl.norm(gB) + _DEF_EPS
    b1 = xK - (FB / (gB_n * gB_n)) * gB

    FB = phi_B(b1)
    gB = csdl.reshape(csdl.derivative(ofs=FB, wrts=b1), (3,))
    gB_n = csdl.norm(gB) + _DEF_EPS
    b = b1 - (FB / (gB_n * gB_n)) * gB


    # ---- outputs ----
    diff = a - b
    gap  = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)  # Euclidean pair gap ≥ 0
    

    if return_all:
        return xK, F_star, a, b, gap
    return F_star






# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     gamma: float = 5.0,                 # penalty strength (try 1–20 depending on units)
#     normalize_grads: bool = True,       # use unit normals to improve conditioning
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_penalty",
#     return_all: bool = False,
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Fully differentiable collision check via the penalized closest-pair stationarity:
#         (a - b) + gamma * phi_A(a) * gA_hat = 0
#         (b - a) + gamma * phi_B(b) * gB_hat = 0
#     Returns pair_gap by default; with return_all=True returns (m, F_star, a, b, pair_gap).
#     """

#     # ---- numeric seeds (leaf states) ----
#     m0 = x0.value.reshape(3,) if isinstance(x0, csdl.Variable) else np.asarray(x0, float).reshape(3,)
#     a0_np = m0.copy()
#     b0_np = m0.copy()

#     a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
#     b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

#     # φ and gradients
#     phiA = phi_A(a)  # scalar
#     phiB = phi_B(b)  # scalar

#     gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))

#     if normalize_grads:
#         gA = gA / csdl.sqrt(csdl.vdot(gA, gA) + _DEF_EPS)
#         gB = gB / csdl.sqrt(csdl.vdot(gB, gB) + _DEF_EPS)

#     # Stationarity residuals (6 equations,3 each)
#     R_a = (a - b) + gamma * phiA * gA   # (3,)
#     R_b = (b - a) + gamma * phiB * gB   # (3,)

#     # Newton solve 
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a, R_a, initial_value=a0_np)
#     solver.add_state(b, R_b, initial_value=b0_np)
#     solver.run()

#     # Outputs
#     diff = a - b
#     pair_gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#     m = 0.5 * (a + b)
#     # keep your scoring-compatible value if you use it downstream
#     F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.0507

#     if return_all:
#         return m, F_star, a, b, pair_gap
#     return F_star

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


# KKT approach (issues with Nan/Inf/Singular Matrix)
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



# # Hybrid method?
# # ----- tiny NumPy helpers (for seeding & projection) -----
# def _fd_grad(phi_np, p, h=1e-5):
#     p = np.asarray(p, float)
#     g = np.zeros(3)
#     for k in range(3):
#         e = np.zeros(3); e[k] = 1.0
#         g[k] = (phi_np(p + h*e) - phi_np(p - h*e)) / (2*h)
#     return g

# def _project_to_surface_np(phi_np, p, iters=12, tol=1e-10):
#     p = np.asarray(p, float).copy()
#     for _ in range(iters):
#         val = float(phi_np(p))
#         g = _fd_grad(phi_np, p)
#         g2 = float(np.dot(g, g)) + _DEF_EPS
#         step = val * g / g2
#         p -= step
#         if abs(val) < tol or np.linalg.norm(step) < tol:
#             break
#     return p

# def _finite(x): return np.all(np.isfinite(x))

# # ----- SAFE collision check: penalty -> (optional) KKT refine -> snap & guards -----
# def collision_check_safe(
#     phi_A, phi_B, x0,
#     *,
#     gamma=5.0,                 # penalty strength
#     use_kkt_refine=True,       # run KKT after penalty
#     normalize_grads_pen=True,  # unit normals in penalty stage
#     normalize_grads_kkt=False, # exact scaling in KKT stage
#     newton_tol=1e-8,
#     jitter=1e-3,               # small seed jitter to avoid kinks/centers
#     return_all=False,
# ):
#     """
#     Always-finite closest-pair finder.
#     Returns pair_gap by default; with return_all=True returns (x, F_star, a, b, pair_gap).
#     """

#     # 0) Make NumPy-callable SDFs for seeding
#     phiA_np = lambda p: float(phi_A(np.asarray(p, float)))
#     phiB_np = lambda p: float(phi_B(np.asarray(p, float)))

#     # 1) Seed: project x0 to both surfaces; jitter slightly to avoid zero-gradient centers
#     x0_np = np.asarray(getattr(x0, "value", x0), float).reshape(3,)
#     a0 = _project_to_surface_np(phiA_np, x0_np)
#     b0 = _project_to_surface_np(phiB_np, x0_np)
#     if jitter > 0:
#         a0 = a0 + jitter * np.random.randn(3)
#         b0 = b0 + jitter * np.random.randn(3)
#     x_init = 0.5*(a0 + b0)

#     # 2) Penalty stage (robust): unit-normal penalized stationarity
#     #    (Your existing penalty solver; shown here calling your function.)
#     try:
#         m_pen, F_pen, a_pen, b_pen, gap_pen = collision_check(
#             phi_A, phi_B, x_init,
#             gamma=gamma,
#             normalize_grads=normalize_grads_pen,
#             newton_tol=newton_tol,
#             return_all=True,
#         )
#         a_use = np.array(a_pen.value if hasattr(a_pen, "value") else a_pen, float).reshape(3,)
#         b_use = np.array(b_pen.value if hasattr(b_pen, "value") else b_pen, float).reshape(3,)
#         x_use = np.array(m_pen.value if hasattr(m_pen, "value") else m_pen, float).reshape(3,)
#     except Exception:
#         # Fallback: just use projected seeds
#         a_use, b_use, x_use = a0, b0, 0.5*(a0+b0)

#     # 3) Optional KKT refine (exact surfaces), warm-started from penalty
#     if use_kkt_refine:
#         try:
#             x_kkt, F_kkt, a_kkt, b_kkt, gap_kkt = collision_check_kkt(
#                 phi_A, phi_B, x_use,
#                 newton_tol=newton_tol,
#                 normalize_grads=normalize_grads_kkt,
#                 return_all=True,
#                 # If your KKT takes custom seeds, pass a_use/b_use in there
#             )
#             a_use = np.array(a_kkt.value if hasattr(a_kkt, "value") else a_kkt, float).reshape(3,)
#             b_use = np.array(b_kkt.value if hasattr(b_kkt, "value") else b_kkt, float).reshape(3,)
#             x_use = np.array(x_kkt.value if hasattr(x_kkt, "value") else x_kkt, float).reshape(3,)
#         except Exception:
#             # keep penalty result if KKT has trouble (tangent, far apart, etc.)
#             pass

#     # 4) Snap outputs to surfaces (cheap, guarantees φ=0 for plotting/consistency)
#     a_surf = _project_to_surface_np(phiA_np, a_use)
#     b_surf = _project_to_surface_np(phiB_np, b_use)
#     gap = float(np.linalg.norm(a_surf - b_surf))
#     F_star = max(phiA_np(x_use), phiB_np(x_use))  # compatibility score at midpoint

#     # 5) Final guarantees: finite values
#     if not (_finite(a_surf) and _finite(b_surf) and np.isfinite(gap) and np.isfinite(F_star)):
#         # As a last resort, rebuild from seeds deterministically
#         a_surf = _project_to_surface_np(phiA_np, x0_np)
#         b_surf = _project_to_surface_np(phiB_np, x0_np)
#         gap = float(np.linalg.norm(a_surf - b_surf))
#         F_star = max(phiA_np(0.5*(a_surf+b_surf)), phiB_np(0.5*(a_surf+b_surf)))

#     if return_all:
#         return x_use, F_star, a_surf, b_surf, gap
#     return gap
