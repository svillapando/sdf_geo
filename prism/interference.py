
from __future__ import annotations
from typing import Callable, Union, Tuple
import numpy as np
import csdl_alpha as csdl


SDF = Callable[[csdl.Variable], csdl.Variable]

_DEF_EPS = 1e-12

# # Collision check similar to paper
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     eta_max: csdl.Variable,
#     *,
#     rho: float = 20.0,             # higher = sharper max; start 5–20
#     return_all: bool = False,
#     newton_name: str = "collision_check_smooth",
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Differentiable SDF-SDF proximity/collision check

#     Steps:
#       1) Define smooth joint field F(x) = max(phi_A(x), phi_B(x)) 
#       2) Do K fixed 'pull' steps: x <- x - eta * ∇F / ||∇F||.
#       3) Project once from x_K to each surface:
#            a = x_K - phi_A(x_K) / ||∇phi_A(x_K)||^2 * ∇phi_A(x_K)
#            b = x_K - phi_B(x_K) / ||∇phi_B(x_K)||^2 * ∇phi_B(x_K)
#       4) Outputs: gap = ||a - b||  (always >= 0), F_star = F(x_K),
#          and optionally x_K, a, b.

#     Returns:
#       If return_all: (x_K, F_star, a, b, gap), else just F.
#     """
#     # ---- seed x0 as leaf variable ----
#     if isinstance(x0, csdl.Variable):
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     else:
#         x0_np = np.asarray(x0, float).reshape(3,)
#     x = csdl.Variable(name=f"{newton_name}:x0", shape=(3,), value=x0_np)

#     # ---- finite gradient steps on F(x) = max(phi_A, phi_B) ----

#     for c_i in (0.9, 0.5, 0.25): 
#     #for c_i in (0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7):

#       # inside each pull iteration
#         FA = phi_A(x)
#         FB = phi_B(x)
#         F  = csdl.maximum(FA, FB, rho=rho)

#         gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))
#         gF_norm = csdl.sqrt(csdl.vdot(gF, gF)) + _DEF_EPS

#         # adaptive step length: eta = min(c * |F|, eta_max)
#         eta = csdl.minimum(c_i * csdl.absolute(F), eta_max)   # choose c_i per pull, e.g. 0.7, 0.35, 0.2

#         x = x - (eta * gF / gF_norm)
 
#     xK = x
#     F_star = csdl.maximum(phi_A(xK), phi_B(xK), rho=rho)


#     #---- two-step projection to each surface from xK ----
#     if return_all:
#         # A
#         FA = phi_A(xK)
#         gA = csdl.reshape(csdl.derivative(ofs=FA, wrts=xK), (3,))
#         gA_n = csdl.norm(gA) + _DEF_EPS
#         a1 = xK - (FA / (gA_n * gA_n)) * gA

#         FA = phi_A(a1)
#         gA = csdl.reshape(csdl.derivative(ofs=FA, wrts=a1), (3,))
#         gA_n = csdl.norm(gA) + _DEF_EPS
#         a = a1 - (FA / (gA_n * gA_n)) * gA

#         # B
#         FB = phi_B(xK)
#         gB = csdl.reshape(csdl.derivative(ofs=FB, wrts=xK), (3,))
#         gB_n = csdl.norm(gB) + _DEF_EPS
#         b1 = xK - (FB / (gB_n * gB_n)) * gB

#         FB = phi_B(b1)
#         gB = csdl.reshape(csdl.derivative(ofs=FB, wrts=b1), (3,))
#         gB_n = csdl.norm(gB) + _DEF_EPS
#         b = b1 - (FB / (gB_n * gB_n)) * gB


#         # ---- outputs ----
#         diff = a - b
#         gap  = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)  # Euclidean pair gap ≥ 0
    
#         # === equal-distance snap of xK, then one reproject ===
#         def sgn_smooth(x, eps=_DEF_EPS):
#         # smooth sign ≈ x / (|x| + eps)
#             return x / (csdl.absolute(x) + eps)

#         # normals at the projected points
#         FAa = phi_A(a); gA_a = csdl.reshape(csdl.derivative(ofs=FAa, wrts=a), (3,))
#         FBb = phi_B(b); gB_b = csdl.reshape(csdl.derivative(ofs=FBb, wrts=b), (3,))
#         na   = gA_a / (csdl.norm(gA_a) + _DEF_EPS)
#         nb   = gB_b / (csdl.norm(gB_b) + _DEF_EPS)

#         # orient normals smoothly to point toward each other across the gap
#         d     = a - b                          # segment from B→A
#         na_o  = sgn_smooth(csdl.vdot(d, na)) * na
#         nb_o  = sgn_smooth(csdl.vdot(-d, nb)) * nb

#         # distances from xK to each surface along the oriented normals
#         tA = csdl.minimum(csdl.vdot(xK - a, na_o))
#         tB = csdl.minimum(csdl.vdot(b  - xK, nb_o))
#         t  = 0.5 * (tA + tB)                   # force equal distance

#         # snap x to the equal-distance point along normals
#         xK = 0.5 * (a + t*na_o + b - t*nb_o)

#         # reproject once from the snapped midpoint (cheap & effective)
#         FA = phi_A(xK); gA = csdl.reshape(csdl.derivative(ofs=FA, wrts=xK), (3,))
#         a  = xK - (FA * gA) / (csdl.vdot(gA, gA) + _DEF_EPS)

#         FB = phi_B(xK); gB = csdl.reshape(csdl.derivative(ofs=FB, wrts=xK), (3,))
#         b  = xK - (FB * gB) / (csdl.vdot(gB, gB) + _DEF_EPS)


#     if return_all:
#         gap = csdl.norm(a - b)
#         return xK, F_star, a, b, gap
#     return F_star, xK





# It works?? Substitute lagrange multipliers
def collision_check(
    phi_A: SDF,
    phi_B: SDF,
    x0: Union[np.ndarray, csdl.Variable],
    *,
    newton_tol: float = 1e-8,
    newton_name: str = "collision_check_stationarity",
    return_all: bool = False,
    normalize_grads: bool = True,
    grad_floor: float = 1e-8,        # guard for ||∇φ||
    reg_pos: float = 1e-6,           # tiny Tikhonov regularization (anchors at seeds)
    surf_weight: float = 1.0,        # penalty strength pushing a,b to their surfaces
    beta_scale_with_grad: bool = True # scale penalty by ||∇φ|| if not normalizing
) -> tuple[csdl.Variable, ...] | csdl.Variable:

    # ----- numeric seeds -----
    try:
        x0_np = np.asarray(x0.value, float).reshape(3,)
    except AttributeError:
        x0_np = np.asarray(x0, float).reshape(3,)

    a0_np = x0_np.copy()
    b0_np = x0_np.copy()

    # constants / guards
    _eps = 1e-12

    # keep copies of the seeds as constants for regularization anchors
    a0_const = csdl.Variable(value=a0_np)
    b0_const = csdl.Variable(value=b0_np)

    ones3 = csdl.Variable(value=np.ones(3))

    # ----- states (only a and b) -----
    a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
    b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

    # ----- SDF values & (safe) normals -----
    # A
    phiA_a = phi_A(a)                                       # scalar
    gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
    if normalize_grads:
        gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
        nA      = gA / gA_norm
        betaA   = surf_weight
    else:
        # still guard the zero-gradient case
        gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
        nA      = gA / gA_norm
        betaA   = surf_weight * (gA_norm if beta_scale_with_grad else 1.0)

    # B
    phiB_b = phi_B(b)                                       # scalar
    gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))
    if normalize_grads:
        gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
        nB      = gB / gB_norm
        betaB   = surf_weight
    else:
        gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
        nB      = gB / gB_norm
        betaB   = surf_weight * (gB_norm if beta_scale_with_grad else 1.0)

    # Reshape scalars for vector scaling
    phiA_as3 = csdl.reshape(phiA_a, (1,)) * ones3
    phiB_as3 = csdl.reshape(phiB_b, (1,)) * ones3

    # ----- residuals (penalized stationarity, no λ/μ states) -----
    # Intuition: original KKT had (a - b) + λ nA = 0, (a - b) - μ nB = 0 with φ_A(a)=0, φ_B(b)=0.
    # Here we replace λ ≈ -β φ_A(a), μ ≈  β φ_B(b), which pushes a,b to their surfaces and couples them.
    RS_A = (a - b) + betaA * (phiA_as3 * nA) + reg_pos * (a - a0_const)  # (3,)
    RS_B = (a - b) - betaB * (phiB_as3 * nB) + reg_pos * (b - b0_const)  # (3,)

    # ----- Newton solve -----
    solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
    # (Optional) If your CSDL build supports it, you can also set a safe max iters, damping, etc.
    # e.g., solver.options['max_iter'] = 80
    solver.add_state(a, RS_A, initial_value=a0_np)   # 3 eqs
    solver.add_state(b, RS_B, initial_value=b0_np)   # 3 eqs
    solver.run()

    # ---------- helpers (shape-safe) ----------
    def _snap_once(p, phi, grad_floor=1e-8):
        # One Newton projection step: p <- p - φ * n, with unit-normal and floor
        val = phi(p)                                              # scalar
        g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
        n   = g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)
        return p - csdl.reshape(val, (1,)) * n                    # (3,)

    def _snap_k(p0, phi, k=2, grad_floor=1e-8):
        p = p0
        for _ in range(k):
            p = _snap_once(p, phi, grad_floor)
        return p

    def _unit(v, eps=1e-12):
        return v / (csdl.sqrt(csdl.vdot(v, v)) + eps)

    def _normal_at(p, phi, grad_floor=1e-8):
        val = phi(p)  # scalar (unused; ensures correct graph deps)
        g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
        return g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)

    # ---------- alternating-projection refinement ----------
    # Idea: land each point on the hemisphere facing the *other* body by projecting
    # from the other body’s point. Do both orders and keep the better one.

    # Option A: start from b toward A, then from a toward B (B→A→B)
    a_A1 = _snap_k(b, phi_A, k=2, grad_floor=grad_floor)         # project B's point onto A
    b_A1 = _snap_k(a_A1, phi_B, k=2, grad_floor=grad_floor)      # then project that onto B

    # Option B: start from a toward B, then from b toward A (A→B→A)
    b_B1 = _snap_k(a, phi_B, k=2, grad_floor=grad_floor)
    a_B1 = _snap_k(b_B1, phi_A, k=2, grad_floor=grad_floor)

    # Score both candidates by (i) smaller gap and (ii) better normal alignment
    def _score_pair(aP, bP, phiA, phiB):
        t   = _unit(aP - bP)
        nA  = _normal_at(aP, phiA, grad_floor)
        nB  = _normal_at(bP, phiB, grad_floor)
        gap = csdl.sqrt(csdl.vdot(aP - bP, aP - bP) + 1e-12)
        # alignment: both outward normals should align with chord (>=0, closer to 1 is better)
        align = 0.5 * (csdl.vdot(nA, t) + csdl.vdot(nB, t))       # scalar
        return gap, align

    gap_A, align_A = _score_pair(a_A1, b_A1, phi_A, phi_B)
    gap_B, align_B = _score_pair(a_B1, b_B1, phi_A, phi_B)

    # Choose the better pair: prefer smaller gap; break ties by larger alignment
    # (CSDL has no conditionals; pick both and *use the one you return* in Python flow)
    use_A = True  # Python-level choice is fine since you're building the graph once per call
    # You can fetch initial seeds’ numpy to decide, but simplest is:
    #   if you want fully CSDL, keep both and return both to caller to choose.
    try:
        # If shapes were numeric Variables, take a quick numpy read for the decision:
        use_A = (float(gap_A.value) < float(gap_B.value)) or \
                (abs(float(gap_A.value) - float(gap_B.value)) < 1e-9 and float(align_A.value) >= float(align_B.value))
    except Exception:
        # Fallback: prefer A by default
        use_A = True

    a_ref = a_A1 if use_A else a_B1
    b_ref = b_A1 if use_A else b_B1

    # # Optional tiny clean-up snap to ensure |φ| ~ 0 after the alternations
    # a_ref = _snap_once(a_ref, phi_A, grad_floor)
    # b_ref = _snap_once(b_ref, phi_B, grad_floor)

    # ---------- replace outputs ----------
    a = a_ref
    b = b_ref
    diff = a - b
    gap  = csdl.sqrt(csdl.vdot(diff, diff) + 1e-12)
    x_mid  = 0.5 * (a + b)
    F_star = csdl.maximum(phi_A(x_mid), phi_B(x_mid), rho=20.0)- 0.07

    if return_all:
        return x_mid, F_star, a, b, gap
    return gap



#IDK ANYMORE KKT AGAIN
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_stationarity",
#     return_all: bool = False,
#     normalize_grads: bool = True,
#     grad_floor: float = 1e-8,        # guard for ||∇φ||
#     reg_pos: float = 1e-6            # tiny Tikhonov regularization in stationarity eqs
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:

#     # ----- numeric seeds -----
#     try:
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     except AttributeError:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np   = x0_np.copy()
#     b0_np   = x0_np.copy()
#     lam0_np = np.array([0.0], dtype=float)
#     mu0_np  = np.array([0.0], dtype=float)

#     # constants / guards
#     _eps = 1e-12 if "_DEF_EPS" in globals() else 1e-12

#     # keep copies of the seeds as constants for regularization anchors
#     a0_const = csdl.Variable(value=a0_np)
#     b0_const = csdl.Variable(value=b0_np)

#     # ----- states -----
#     a   = csdl.Variable(name=f"{newton_name}:a",   shape=(3,), value=a0_np)
#     b   = csdl.Variable(name=f"{newton_name}:b",   shape=(3,), value=b0_np)
#     lam = csdl.Variable(name=f"{newton_name}:lam", shape=(1,), value=lam0_np)
#     mu  = csdl.Variable(name=f"{newton_name}:mu",  shape=(1,), value=mu0_np)

#     # ----- SDF values & (safe) normals -----
#     # A
#     phiA_a = phi_A(a)                                        # scalar
#     gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     if normalize_grads:
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#     else:
#         # still guard the zero-gradient case
#         scale   = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / scale

#     # B
#     phiB_b = phi_B(b)                                        # scalar
#     gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))
#     if normalize_grads:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#     else:
#         scale   = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / scale

#     # broadcast λ, μ to 3-vectors
#     ones3 = csdl.Variable(value=np.ones(3))
#     lam3  = csdl.reshape(csdl.tensordot(lam, ones3, axes=None), (3,))
#     mu3   = csdl.reshape(csdl.tensordot(mu,  ones3, axes=None), (3,))

#     # ----- residuals -----
#     # Surface constraints
#     RA = csdl.reshape(phiA_a, (1,))                    # φ_A(a) = 0
#     RB = csdl.reshape(phiB_b, (1,))                    # φ_B(b) = 0

#     # Stationarity with tiny Tikhonov (anchors at seeds)
#     #   (a - b) + λ nA + reg*(a - a0) = 0
#     #   (a - b) - μ nB + reg*(b - b0) = 0
#     RS_A = (a - b) + lam3 * nA + reg_pos * (a - a0_const)   # (3,)
#     RS_B = (a - b) - mu3  * nB + reg_pos * (b - b0_const)   # (3,)

#     # ----- Newton solve -----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a,   RS_A, initial_value=a0_np)   # 3 eqs
#     solver.add_state(b,   RS_B, initial_value=b0_np)   # 3 eqs
#     solver.add_state(lam, RA,   initial_value=lam0_np) # 1 eq
#     solver.add_state(mu,  RB,   initial_value=mu0_np)  # 1 eq
#     solver.run()

#     # ----- outputs -----
#     diff = a - b
#     gap  = csdl.sqrt(csdl.vdot(diff, diff) + _eps)

#     # mid-point & smooth max (kept for API compatibility)
#     x_mid  = 0.5 * (a + b)
#     F_star = csdl.maximum(phi_A(x_mid), phi_B(x_mid), rho=20.0)

#     if return_all:
#         return x_mid, F_star, a, b, gap
#     return gap











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
def collision_check_kkt_LM(
    phi_A: SDF,
    phi_B: SDF,
    x0: Union[np.ndarray, csdl.Variable],
    *,
    max_iter: int = 60,
    tol: float = 1e-10,
    mu0: float = 1e-2,       # initial LM damping
    mu_inc: float = 10.0,    # if step rejected: mu <- mu * mu_inc
    mu_dec: float = 1/3.0,   # if step accepted: mu <- max(mu*mu_dec, 1e-15)
    return_all: bool = False,
):
    """Levenberg–Marquardt on the textbook KKT residuals (8 unknowns).
       Unknowns: a(3), b(3), lamA(1), lamB(1)
       Residuals:
         R1 = (a - b) + lamA * grad(phi_A)(a)      (3)
         R2 = -(a - b) + lamB * grad(phi_B)(b)     (3)
         R3 = phi_A(a)                              (1)
         R4 = phi_B(b)                              (1)
    """
    # ---- numeric seeds ----
    try:
        x0_np = np.asarray(x0.value, float).reshape(3,)
    except AttributeError:
        x0_np = np.asarray(x0, float).reshape(3,)
    a = x0_np.copy()
    b = x0_np.copy()
    lamA = np.array([0.0], dtype=float)
    lamB = np.array([0.0], dtype=float)

    mu = float(mu0)

    # Basis vectors to extract residual components without array indexing
    def basis(i):
        e = np.zeros(3, dtype=float); e[i] = 1.0
        return csdl.Variable(value=e)

    def eval_F_and_J(a_np, b_np, lamA_np, lamB_np):
        """Build residuals in CSDL, then pull numeric F and J row-by-row."""
        # Wrap current state as Variables
        a_v    = csdl.Variable(value=a_np.reshape(3,))
        b_v    = csdl.Variable(value=b_np.reshape(3,))
        lamA_v = csdl.Variable(value=lamA_np.reshape(1,))
        lamB_v = csdl.Variable(value=lamB_np.reshape(1,))

        # SDF values and gradients
        phiA_a = phi_A(a_v)  # scalar
        gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a_v), (3,))
        phiB_b = phi_B(b_v)  # scalar
        gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b_v), (3,))

        diff = a_v - b_v  # (3,)

        R1 = diff + csdl.reshape(lamA_v, (1,)) * gA   # (3,)
        R2 = -diff + csdl.reshape(lamB_v, (1,)) * gB  # (3,)
        R3 = phiA_a                                   # (1,)
        R4 = phiB_b                                   # (1,)

        # Assemble numeric residual vector F (8,)
        F = np.zeros(8, dtype=float)
        # R1 components
        for i in range(3):
            ri = csdl.vdot(basis(i), R1)
            F[i] = float(ri.value)
        # R2 components
        for i in range(3):
            ri = csdl.vdot(basis(i), R2)
            F[3 + i] = float(ri.value)
        # R3, R4
        F[6] = float(R3.value)
        F[7] = float(R4.value)

        # Build Jacobian J (8x8) wrt [a(3), b(3), lamA, lamB]
        J = np.zeros((8, 8), dtype=float)

        # Helper to append gradients for a scalar residual r
        def fill_row(row_idx, r_scalar):
            da = csdl.derivative(ofs=r_scalar, wrts=a_v)               # (3,)
            db = csdl.derivative(ofs=r_scalar, wrts=b_v)               # (3,)
            dlamA = csdl.derivative(ofs=r_scalar, wrts=lamA_v)         # (1,)
            dlamB = csdl.derivative(ofs=r_scalar, wrts=lamB_v)         # (1,)
            J[row_idx, 0:3] = np.asarray(da.value, float).reshape(3,)
            J[row_idx, 3:6] = np.asarray(db.value, float).reshape(3,)
            J[row_idx, 6]   = float(dlamA.value)
            J[row_idx, 7]   = float(dlamB.value)

        # R1 rows
        for i in range(3):
            ri = csdl.vdot(basis(i), R1)
            fill_row(i, ri)
        # R2 rows
        for i in range(3):
            ri = csdl.vdot(basis(i), R2)
            fill_row(3 + i, ri)
        # R3, R4 rows
        fill_row(6, R3)
        fill_row(7, R4)

        return F, J

    # Initial residual & norm
    F, J = eval_F_and_J(a, b, lamA, lamB)
    f2 = float(F @ F)

    for it in range(max_iter):
        # LM step: (J^T J + mu I) Δ = -J^T F
        JTJ = J.T @ J
        g   = J.T @ F
        # Damping
        JTJ_damped = JTJ + mu * np.eye(8, dtype=float)

        # Solve for Δ; if singular, bump mu and retry
        try:
            delta = -np.linalg.solve(JTJ_damped, g)
        except np.linalg.LinAlgError:
            mu *= mu_inc
            continue

        # Trial update
        a_trial    = a    + delta[0:3]
        b_trial    = b    + delta[3:6]
        lamA_trial = lamA + delta[6:7]
        lamB_trial = lamB + delta[7:8]

        F_trial, J_trial = eval_F_and_J(a_trial, b_trial, lamA_trial, lamB_trial)
        f2_trial = float(F_trial @ F_trial)

        # Accept/reject
        if f2_trial < f2:  # success
            a, b, lamA, lamB = a_trial, b_trial, lamA_trial, lamB_trial
            F, J, f2 = F_trial, J_trial, f2_trial
            mu = max(mu * mu_dec, 1e-15)
            # Convergence?
            if f2 < tol:
                break
        else:
            mu *= mu_inc  # reject step, increase damping and try again

    # Outputs
    a_v = csdl.Variable(value=a.reshape(3,))
    b_v = csdl.Variable(value=b.reshape(3,))
    diff = a_v - b_v
    gap  = csdl.sqrt(csdl.vdot(diff, diff))
    x_mid = 0.5 * (a_v + b_v)

    if return_all:
        lamA_v = csdl.Variable(value=lamA.reshape(1,))
        lamB_v = csdl.Variable(value=lamB.reshape(1,))
        return x_mid, a_v, b_v, lamA_v, lamB_v, gap
    return gap
