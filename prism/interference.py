
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
    rho: float = 20.0,             # higher = sharper max; start 5–20
    return_all: bool = False,
    newton_name: str = "collision_check_smooth",
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

    #for c_i in (0.9, 0.5, 0.25): 
    c_i = 1.0

      # inside each pull iteration
    FA = phi_A(x)
    FB = phi_B(x)
    F  = csdl.maximum(FA, FB, rho=rho)

    gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))
    gF_norm = csdl.sqrt(csdl.vdot(gF, gF)) + _DEF_EPS

    # adaptive step length: eta = min(c * |F|, eta_max)
    eta = csdl.minimum(c_i * csdl.absolute(F), eta_max)   # choose c_i per pull, e.g. 0.7, 0.35, 0.2

    x = x - (eta * gF / gF_norm)

    xK = x
    F_star = csdl.maximum(phi_A(xK), phi_B(xK), rho=rho) -1.5

    
    #---- two-step projection to each surface from xK ----
    if return_all:
        # ---------- helpers: safe L2 norm and unit ----------
        def safe_norm2(v, eps=_DEF_EPS):
            ax = csdl.absolute(v)
            vmax01 = csdl.maximum(ax[0], ax[1])
            vmax   = csdl.maximum(vmax01, ax[2])            # max(|v_i|)
            scale  = csdl.maximum(vmax, csdl.Variable(value = 1.0))                # keep >= 1 to avoid div-by-zero
            v_sc   = v / scale
            return scale * csdl.sqrt(csdl.vdot(v_sc, v_sc) + eps)

        def safe_unit(v, eps=_DEF_EPS):
            return v / (safe_norm2(v, eps) + eps)

        # ---- keep a copy of the candidate x* for residuals ----
        x_cand = xK
        np.set_printoptions(precision=6, suppress=True)

        def _inf_norm(u):      # for numpy arrays (after .value)
            return float(np.max(np.abs(u)))


        x_cand = xK

        # ===== A: first projection step =====
        print("[PIN] before phi_A(x)")
        FA0 = phi_A(x_cand)
        print("[PIN] after  phi_A(x) | FA(x)=", float(FA0.value))

        print("[PIN] before gradA(x)")
        gA0 = csdl.reshape(csdl.derivative(ofs=FA0, wrts=x_cand), (3,))
        print("[PIN] after  gradA(x) | max|gA|=", float(np.max(np.abs(gA0.value))))

        gA0n = safe_norm2(gA0) + _DEF_EPS
        sA   = csdl.absolute(FA0) / gA0n
        a1   = x_cand - (FA0 * gA0) / (gA0n * gA0n)

        print("[DBG] A0  | FA(x)       =", float(FA0.value),
                "||gA||    =", float(gA0n.value),
                "max|gA|   =", float(np.max(np.abs(gA0.value))),
                "step      =", float(sA.value),
                "||x||inf  =", _inf_norm(x_cand.value),
                "||a1||inf =", _inf_norm(a1.value))

        # This call is a common overflow point if a1 jumped far:
        FA1 = phi_A(a1)
        print("[DBG] A1  | FA(a1)      =", float(FA1.value))

        # ===== A: second projection step =====
        gA1  = csdl.reshape(csdl.derivative(ofs=FA1, wrts=a1), (3,))
        gA1n = safe_norm2(gA1) + _DEF_EPS
        a    = a1 - (FA1 * gA1) / (gA1n * gA1n)

        print("[DBG] A2  | ||gA(a1)||  =", float(gA1n.value),
                "max|gA(a1)|=", float(np.max(np.abs(gA1.value))),
                "||a||inf   =", _inf_norm(a.value))

        # ===== B: first projection step =====
        FB0  = phi_B(x_cand)
        gB0  = csdl.reshape(csdl.derivative(ofs=FB0, wrts=x_cand), (3,))
        gB0n = safe_norm2(gB0) + _DEF_EPS
        sB   = csdl.absolute(FB0) / gB0n
        b1   = x_cand - (FB0 * gB0) / (gB0n * gB0n)

        print("[DBG] B0  | FB(x)       =", float(FB0.value),
                "||gB||    =", float(gB0n.value),
                "max|gB|   =", float(np.max(np.abs(gB0.value))),
                "step      =", float(sB.value),
                "||x||inf  =", _inf_norm(x_cand.value),
                "||b1||inf =", _inf_norm(b1.value))

        # Another common overflow point:
        FB1 = phi_B(b1)
        print("[DBG] B1  | FB(b1)      =", float(FB1.value))

        # ===== B: second projection step =====
        gB1  = csdl.reshape(csdl.derivative(ofs=FB1, wrts=b1), (3,))
        gB1n = safe_norm2(gB1) + _DEF_EPS
        b    = b1 - (FB1 * gB1) / (gB1n * gB1n)

        print("[DBG] B2  | ||gB(b1)||  =", float(gB1n.value),
                "max|gB(b1)|=", float(np.max(np.abs(gB1.value))),
                "||b||inf   =", _inf_norm(b.value))

        # ---- gap (optional) ----
        diff = a - b
        gap  = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)
        print("[DBG] GAP | ||a-b||     =", float(gap.value))

        # ===== residuals =====
        FA_x = phi_A(x_cand); FB_x = phi_B(x_cand)
        r_eq = csdl.absolute(FA_x - FB_x)
        print("[DBG] EQ  | |FA-FB|     =", float(r_eq.value))

        FA_a = phi_A(a); gA_a = csdl.reshape(csdl.derivative(ofs=FA_a, wrts=a), (3,))
        FB_b = phi_B(b); gB_b = csdl.reshape(csdl.derivative(ofs=FB_b, wrts=b), (3,))
        nA   = gA_a / (safe_norm2(gA_a) + _DEF_EPS)
        nB   = gB_b / (safe_norm2(gB_b) + _DEF_EPS)

        print("[DBG] NRM | ||gA(a)||   =", float(safe_norm2(gA_a).value),
                "||gB(b)|| =", float(safe_norm2(gB_b).value),
                "max|gA(a)|=", float(np.max(np.abs(gA_a.value))),
                "max|gB(b)|=", float(np.max(np.abs(gB_b.value))))

        r_dir = 0.5 * safe_norm2(nA + nB)
        print("[DBG] DIR | r_dir       =", float(r_dir.value))

        r_eik_A = csdl.absolute(safe_norm2(gA_a) - 1.0)
        r_eik_B = csdl.absolute(safe_norm2(gB_b) - 1.0)
        print("[DBG] EIK | A=", float(r_eik_A.value), "B=", float(r_eik_B.value))


    if return_all:
        return xK, F_star, a, b
    return xK, F_star





# # It works?? Substitute lagrange multipliers
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
#     reg_pos: float = 1e-6,           # tiny Tikhonov regularization (anchors at seeds)
#     surf_weight: float = 1.0,        # penalty strength pushing a,b to their surfaces
#     beta_scale_with_grad: bool = True # scale penalty by ||∇φ|| if not normalizing
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:

#     # ----- numeric seeds -----
#     try:
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     except AttributeError:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np = x0_np.copy()
#     b0_np = x0_np.copy()

#     # constants / guards
#     _eps = 1e-12

#     # keep copies of the seeds as constants for regularization anchors
#     a0_const = csdl.Variable(value=a0_np)
#     b0_const = csdl.Variable(value=b0_np)

#     ones3 = csdl.Variable(value=np.ones(3))

#     # ----- states (only a and b) -----
#     a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
#     b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

#     # ----- SDF values & (safe) normals -----
#     # A
#     phiA_a = phi_A(a)                                       # scalar
#     gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     if normalize_grads:
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#         betaA   = surf_weight
#     else:
#         # still guard the zero-gradient case
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#         betaA   = surf_weight * (gA_norm if beta_scale_with_grad else 1.0)

#     # B
#     phiB_b = phi_B(b)                                       # scalar
#     gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))
#     if normalize_grads:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#         betaB   = surf_weight
#     else:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#         betaB   = surf_weight * (gB_norm if beta_scale_with_grad else 1.0)

#     # Reshape scalars for vector scaling
#     phiA_as3 = csdl.reshape(phiA_a, (1,)) * ones3
#     phiB_as3 = csdl.reshape(phiB_b, (1,)) * ones3

#     # ----- residuals (penalized stationarity, no λ/μ states) -----
#     # Intuition: original KKT had (a - b) + λ nA = 0, (a - b) - μ nB = 0 with φ_A(a)=0, φ_B(b)=0.
#     # Here we replace λ ≈ -β φ_A(a), μ ≈  β φ_B(b), which pushes a,b to their surfaces and couples them.
#     RS_A = (a - b) + betaA * (phiA_as3 * nA) + reg_pos * (a - a0_const)  # (3,)
#     RS_B = (a - b) - betaB * (phiB_as3 * nB) + reg_pos * (b - b0_const)  # (3,)

#     # ----- Newton solve -----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     # (Optional) If your CSDL build supports it, you can also set a safe max iters, damping, etc.
#     # e.g., solver.options['max_iter'] = 80
#     solver.add_state(a, RS_A, initial_value=a0_np)   # 3 eqs
#     solver.add_state(b, RS_B, initial_value=b0_np)   # 3 eqs
#     solver.run()

#     # ---------- helpers (shape-safe) ----------
#     def _snap_once(p, phi, grad_floor=1e-8):
#         # One Newton projection step: p <- p - φ * n, with unit-normal and floor
#         val = phi(p)                                              # scalar
#         g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
#         n   = g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)
#         return p - csdl.reshape(val, (1,)) * n                    # (3,)

#     def _snap_k(p0, phi, k=2, grad_floor=1e-8):
#         p = p0
#         for _ in range(k):
#             p = _snap_once(p, phi, grad_floor)
#         return p

#     def _unit(v, eps=1e-12):
#         return v / (csdl.sqrt(csdl.vdot(v, v)) + eps)

#     def _normal_at(p, phi, grad_floor=1e-8):
#         val = phi(p)  # scalar (unused; ensures correct graph deps)
#         g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
#         return g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)

#     # ---------- alternating-projection refinement ----------
#     # Idea: land each point on the hemisphere facing the *other* body by projecting
#     # from the other body’s point. Do both orders and keep the better one.

#     # Option A: start from b toward A, then from a toward B (B→A→B)
#     a_A1 = _snap_k(b, phi_A, k=2, grad_floor=grad_floor)         # project B's point onto A
#     b_A1 = _snap_k(a_A1, phi_B, k=2, grad_floor=grad_floor)      # then project that onto B

#     # Option B: start from a toward B, then from b toward A (A→B→A)
#     b_B1 = _snap_k(a, phi_B, k=2, grad_floor=grad_floor)
#     a_B1 = _snap_k(b_B1, phi_A, k=2, grad_floor=grad_floor)

#     # Score both candidates by (i) smaller gap and (ii) better normal alignment
#     def _score_pair(aP, bP, phiA, phiB):
#         t   = _unit(aP - bP)
#         nA  = _normal_at(aP, phiA, grad_floor)
#         nB  = _normal_at(bP, phiB, grad_floor)
#         gap = csdl.sqrt(csdl.vdot(aP - bP, aP - bP) + 1e-12)
#         # alignment: both outward normals should align with chord (>=0, closer to 1 is better)
#         align = 0.5 * (csdl.vdot(nA, t) + csdl.vdot(nB, t))       # scalar
#         return gap, align

#     gap_A, align_A = _score_pair(a_A1, b_A1, phi_A, phi_B)
#     gap_B, align_B = _score_pair(a_B1, b_B1, phi_A, phi_B)

#     # Choose the better pair: prefer smaller gap; break ties by larger alignment
#     # (CSDL has no conditionals; pick both and *use the one you return* in Python flow)
#     use_A = True  # Python-level choice is fine since you're building the graph once per call
#     # You can fetch initial seeds’ numpy to decide, but simplest is:
#     #   if you want fully CSDL, keep both and return both to caller to choose.
#     try:
#         # If shapes were numeric Variables, take a quick numpy read for the decision:
#         use_A = (float(gap_A.value) < float(gap_B.value)) or \
#                 (abs(float(gap_A.value) - float(gap_B.value)) < 1e-9 and float(align_A.value) >= float(align_B.value))
#     except Exception:
#         # Fallback: prefer A by default
#         use_A = True

#     a_ref = a_A1 if use_A else a_B1
#     b_ref = b_A1 if use_A else b_B1

#     # # Optional tiny clean-up snap to ensure |φ| ~ 0 after the alternations
#     # a_ref = _snap_once(a_ref, phi_A, grad_floor)
#     # b_ref = _snap_once(b_ref, phi_B, grad_floor)

#     # ---------- replace outputs ----------
#     a = a_ref
#     b = b_ref
#     diff = a - b
#     gap  = csdl.sqrt(csdl.vdot(diff, diff) + 1e-12)
#     x_mid  = 0.5 * (a + b)
#     F_star = csdl.maximum(phi_A(x_mid), phi_B(x_mid), rho=20.0)- 0.07

#     if return_all:
#         return x_mid, F_star, a, b, gap
#     return gap



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

# # ------------------------------------------------------------
# # Two-phase collision solve with feasibility weighting
# # Phase 1: LM projection to each surface (gets |phi| ~ 0)
# # Phase 2: KKT LM with weighted feasibility (robust from far seeds)
# # ------------------------------------------------------------

# def collision_check_kkt_LM(
#     phi_A, phi_B, x0,
#     *,
#     # Phase-1 (projection) LM settings
#     proj_max_iter: int = 40,
#     proj_tol: float = 1e-12,
#     # Phase-2 (KKT) LM settings
#     kkt_max_iter: int = 80,
#     kkt_tol: float = 1e-12,
#     lm_mu0: float = 1e-2,
#     lm_mu_inc: float = 10.0,
#     lm_mu_dec: float = 1/3,
#     lm_eta: float = 1e-3,             # gain ratio acceptance threshold
#     # Feasibility weighting schedule (for Phase-2)
#     feas_weight_init: float = 50.0,    # start heavy so φ-terms dominate when far
#     feas_weight_decay: float = 0.5,    # multiply weight by this each schedule step
#     feas_taper_threshold: float = 1e-3,# once max(|φ|) below this, set weight=1
#     feas_schedule_steps: int = 3,      # how many weighted passes before w=1
#     # Misc
#     return_all: bool = False
# ):
#     """
#     Returns: gap (default)
#              or (x_mid, a, b, gap) if return_all=True
#     """

#     # ---------- utils to extract numpy ----------
#     def _to_np(x, shape):
#         try:
#             return np.asarray(x.value, float).reshape(*shape)
#         except Exception:
#             return np.asarray(x, float).reshape(*shape)

#     # ---------- Phase-1: LM projection to a surface ----------
#     def _lm_project(phi, p0_np):
#         mu = float(lm_mu0)
#         p = p0_np.copy()

#         def FJ(p_np):
#             p_v = csdl.Variable(value=p_np.reshape(3,))
#             val = phi(p_v)                                        # scalar
#             g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p_v),(3,))
#             F   = np.array([float(val.value)], float)             # (1,)
#             J   = np.zeros((1,3), float); J[0,:] = _to_np(g,(3,))
#             return F, J

#         F, J = FJ(p); f2 = float(F @ F)
#         for _ in range(proj_max_iter):
#             JTJ = J.T @ J
#             g = J.T @ F
#             try:
#                 dp = -np.linalg.solve(JTJ + mu*np.eye(3), g).reshape(3,)
#             except np.linalg.LinAlgError:
#                 mu *= lm_mu_inc; continue
#             p_trial = p + dp
#             F_trial, J_trial = FJ(p_trial); f2_trial = float(F_trial @ F_trial)

#             # predicted reduction (quadratic model on 1D residual)
#             pred_red = - (dp @ g).item() - 0.5 * (dp @ (JTJ @ dp)).item()
#             rho = (f2 - f2_trial) / max(pred_red, 1e-16)

#             if rho > lm_eta and f2_trial < f2:
#                 p, F, J, f2 = p_trial, F_trial, J_trial, f2_trial
#                 mu = max(mu * lm_mu_dec, 1e-15)
#                 if f2 < proj_tol: break
#             else:
#                 mu *= lm_mu_inc
#         return p

#     # ---------- Phase-2: weighted KKT LM (one run) ----------
#     # Unknowns: a(3), b(3), lamA(1), lamB(1) -> x in R^8
#     def _kkt_LM(a_seed_np, b_seed_np, w_phi: float, max_iter: int, tol: float):
#         def pack(a, b, lamA, lamB):
#             x = np.zeros(8, float)
#             x[0:3] = a; x[3:6] = b; x[6] = lamA[0]; x[7] = lamB[0]
#             return x
#         def unpack(x):
#             return x[0:3], x[3:6], np.array([x[6]]), np.array([x[7]])

#         mu = float(lm_mu0)
#         x = pack(a_seed_np, b_seed_np, np.array([0.0]), np.array([0.0]))

#         def FJ(x_vec):
#             a_np, b_np, lamA_np, lamB_np = unpack(x_vec)

#             a_v = csdl.Variable(value=a_np.reshape(3,))
#             b_v = csdl.Variable(value=b_np.reshape(3,))
#             lamA_v = csdl.Variable(value=lamA_np.reshape(1,))
#             lamB_v = csdl.Variable(value=lamB_np.reshape(1,))

#             # SDFs and gradients
#             phiA_a = phi_A(a_v); gA = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a_v), (3,))
#             phiB_b = phi_B(b_v); gB = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b_v), (3,))

#             diff = a_v - b_v

#             # Residuals:
#             # R1 = (a-b) + lamA * grad(phi_A)(a)    (3)
#             # R2 = -(a-b) + lamB * grad(phi_B)(b)   (3)
#             # R3 = w_phi * phi_A(a)                 (1)
#             # R4 = w_phi * phi_B(b)                 (1)
#             R1 = diff + csdl.reshape(lamA_v,(1,)) * gA
#             R2 = -diff + csdl.reshape(lamB_v,(1,)) * gB
#             R3 = w_phi * phiA_a
#             R4 = w_phi * phiB_b

#             F = np.zeros(8, float)
#             F[0:3] = _to_np(R1,(3,))
#             F[3:6] = _to_np(R2,(3,))
#             F[6]   = float(R3.value)
#             F[7]   = float(R4.value)

#             # Jacobian via AD row-by-row
#             J = np.zeros((8,8), float)

#             def fill_row(i, r_scalar):
#                 da = csdl.derivative(ofs=r_scalar, wrts=a_v)  # (3,)
#                 db = csdl.derivative(ofs=r_scalar, wrts=b_v)  # (3,)
#                 dA = csdl.derivative(ofs=r_scalar, wrts=lamA_v) # (1,)
#                 dB = csdl.derivative(ofs=r_scalar, wrts=lamB_v) # (1,)
#                 J[i,0:3] = _to_np(da,(3,))
#                 J[i,3:6] = _to_np(db,(3,))
#                 J[i,6]   = float(dA.value)
#                 J[i,7]   = float(dB.value)

#             # R1 rows
#             fill_row(0, R1[0]); fill_row(1, R1[1]); fill_row(2, R1[2])
#             # R2 rows
#             fill_row(3, R2[0]); fill_row(4, R2[1]); fill_row(5, R2[2])
#             # R3, R4 rows
#             fill_row(6, R3); fill_row(7, R4)

#             # Also return raw φ for taper decisions
#             return F, J, float(phiA_a.value), float(phiB_b.value)

#         F, J, phiA_val, phiB_val = FJ(x); f2 = float(F @ F)
#         for _ in range(max_iter):
#             JTJ = J.T @ J
#             g = J.T @ F
#             try:
#                 dx = -np.linalg.solve(JTJ + mu*np.eye(8), g)
#             except np.linalg.LinAlgError:
#                 mu *= lm_mu_inc; continue

#             x_trial = x + dx
#             F_trial, J_trial, phiA_t, phiB_t = FJ(x_trial)
#             f2_trial = float(F_trial @ F_trial)

#             # Predicted reduction (Gauss–Newton model)
#             pred_red = - (dx @ g) - 0.5 * (dx @ (JTJ @ dx))
#             rho = (f2 - f2_trial) / max(pred_red, 1e-16)

#             if rho > lm_eta and f2_trial < f2:
#                 x, F, J, f2 = x_trial, F_trial, J_trial, f2_trial
#                 phiA_val, phiB_val = phiA_t, phiB_t
#                 mu = max(mu * lm_mu_dec, 1e-15)
#                 if f2 < kkt_tol:
#                     break
#             else:
#                 mu *= lm_mu_inc

#         a_np, b_np, lamA_np, lamB_np = unpack(x)
#         return a_np, b_np, lamA_np, lamB_np, f2, max(abs(phiA_val), abs(phiB_val))

#     # ---------- Build Phase-1 seeds (two orders), run Phase-2 with schedule ----------
#     x0_np = _to_np(x0, (3,))

#     # Order A→B
#     a_seed_A = _lm_project(phi_A, x0_np)
#     b_seed_A = _lm_project(phi_B, a_seed_A)

#     # Order B→A
#     b_seed_B = _lm_project(phi_B, x0_np)
#     a_seed_B = _lm_project(phi_A, b_seed_B)

#     def run_schedule(a0_np, b0_np):
#         a_np, b_np = a0_np.copy(), b0_np.copy()
#         # Feasibility-weighted passes
#         w = max(1.0, float(feas_weight_init))
#         for s in range(max(1, feas_schedule_steps)):
#             a_np, b_np, lamA_np, lamB_np, f2, feas = _kkt_LM(
#                 a_np, b_np, w_phi=w, max_iter=kkt_max_iter, tol=kkt_tol
#             )
#             # Taper weight if feasible enough
#             if feas <= feas_taper_threshold:
#                 w = 1.0
#                 break
#             w = max(1.0, w * float(feas_weight_decay))
#         # Final refinement at weight=1 (ensures unbiased stationarity)
#         a_np, b_np, lamA_np, lamB_np, f2, feas = _kkt_LM(
#             a_np, b_np, w_phi=1.0, max_iter=kkt_max_iter, tol=kkt_tol
#         )
#         return a_np, b_np, f2, feas

#     aA_np, bA_np, f2A, feasA = run_schedule(a_seed_A, b_seed_A)
#     aB_np, bB_np, f2B, feasB = run_schedule(a_seed_B, b_seed_B)

#     # Pick the better candidate (prefer smaller gap; tie-break by feasibility & f2)
#     def gap_of(a_np, b_np):
#         diff = a_np - b_np
#         return float(np.sqrt(np.dot(diff, diff)))

#     gapA = gap_of(aA_np, bA_np)
#     gapB = gap_of(aB_np, bB_np)

#     if (gapA < gapB) or (abs(gapA-gapB) < 1e-12 and (feasA, f2A) <= (feasB, f2B)):
#         a_best, b_best, gap_best = aA_np, bA_np, gapA
#     else:
#         a_best, b_best, gap_best = aB_np, bB_np, gapB

#     # ---- Pack outputs as CSDL Variables (for downstream graph use) ----
#     a_v = csdl.Variable(value=a_best.reshape(3,))
#     b_v = csdl.Variable(value=b_best.reshape(3,))
#     diff_v = a_v - b_v
#     gap_v = csdl.sqrt(csdl.vdot(diff_v, diff_v) + 1e-16)
#     x_mid = 0.5 * (a_v + b_v)

#     if return_all:
#         return x_mid, a_v, b_v, gap_v
#     return gap_v
