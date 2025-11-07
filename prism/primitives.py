import numpy as np
import csdl_alpha as csdl
from scipy.spatial.transform import Rotation


_EPS = 1e-12

def sdf_box(center, half_size, rotation_angles, degrees=True):
    def _sdf(p):
        R = Rotation.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()
        R_T = R.T


        # ---- broadcast geometry with CSDL-safe ops ----
        if len(p.shape) == 1:
            center_bcast = center
            half_bcast   = half_size
        else:
            lead_shape   = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            center_bcast = csdl.tensordot(ones_leading, center, axes=None)
            half_bcast   = csdl.tensordot(ones_leading, half_size, axes=None)

        # ---- rotate into box frame ----
        p_local = p - center_bcast
        last_axis = len(p_local.shape) - 1
        p_rot = csdl.tensordot(p_local, R_T, axes=([last_axis], [1]))

        # ---- box SDF with safe norms ----
        q = csdl.absolute(p_rot) - half_bcast
        q_clip = csdl.maximum(q, np.broadcast_to(0.0, q.shape))

        if len(q_clip.shape) == 1:
            # ||q_clip||_2 with _EPSilon (safe)
            norm_sq = csdl.vdot(q_clip, q_clip)
            norm_val = csdl.sqrt(norm_sq + _EPS)
        else:
            # sum over last axis
            norm_sq = csdl.sum(q_clip * q_clip, axes=(len(q_clip.shape) - 1,))
            norm_val = csdl.sqrt(norm_sq + _EPS)

        q_max = csdl.maximum(q, axes=(len(q.shape) - 1,))
        correction = csdl.minimum(q_max, np.broadcast_to(0.0, q_max.shape))

        return norm_val + correction

    return _sdf


def sdf_sphere(center, radius):
    def _sdf(p):
  

        # ---- broadcast center with CSDL-safe ops ----
        if len(p.shape) == 1:
            center_bcast = center
        else:
            lead_shape   = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            center_bcast = csdl.tensordot(ones_leading, center, axes=None)

        dist = p - center_bcast
        last_axis = len(p.shape) - 1

        if len(p.shape) == 1:
            # safe Euclidean norm
            norm_sq = csdl.vdot(dist, dist)
            return csdl.sqrt(norm_sq + _EPS) - radius
        else:
            norm_sq = csdl.sum(dist * dist, axes=(last_axis,))
            return csdl.sqrt(norm_sq + _EPS) - radius

    return _sdf


def sdf_plane(p0, normal):
    def _sdf(p):

        # ---- normalize normal (true signed distance scale) ----
        n_norm = csdl.sqrt(csdl.vdot(normal, normal) + _EPS)
        n_unit = normal / n_norm

        # ---- broadcast with CSDL-safe ops ----
        if len(p.shape) == 1:
            p0_b = p0
            n_b  = n_unit
            delta = p - p0_b
            return csdl.vdot(delta, n_b)
        else:
            lead_shape   = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            p0_b = csdl.tensordot(ones_leading, p0, axes=None)
            n_b  = csdl.tensordot(ones_leading, n_unit, axes=None)
            delta = p - p0_b
            return csdl.sum(delta * n_b, axes=(len(p.shape) - 1,))
    return _sdf


def sdf_capsule(p1, p2, radius):
    def _sdf(p):

        # ---- broadcast p1 with CSDL-safe ops ----
        if len(p.shape) == 1:
            p1_b = p1
        else:
            lead_shape   = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            p1_b = csdl.tensordot(ones_leading, p1, axes=None)

        ba = p2 - p1                          # (3,)
        axis_len_sq = csdl.sum(ba * ba) + _EPS # add _EPS before division

        pa = p - p1_b                          # (..., 3)
        last = len(p.shape) - 1

        # t = ((p-p1)·ba) / (ba·ba)
        t_numer = csdl.tensordot(pa, ba, axes=([last], [0]))  # shape (...,)
        t = t_numer / axis_len_sq

        # clamp t to [0,1]
        t = csdl.minimum(
                csdl.maximum(t, np.broadcast_to(0.0, t.shape)),
                np.broadcast_to(1.0, t.shape)
            )

        # proj = p1 + t * ba  (broadcasted safely)
        proj = p1_b + csdl.reshape(csdl.tensordot(t, ba, axes=None), p.shape)

        # distance to segment with safe norm
        if len(p.shape) == 1:
            diff = p - proj
            dist_sq = csdl.vdot(diff, diff)
            d = csdl.sqrt(dist_sq + _EPS) - radius
        else:
            diff = p - proj
            dist_sq = csdl.sum(diff * diff, axes=(len(p.shape) - 1,))
            d = csdl.sqrt(dist_sq + _EPS) - radius
        return d

    return _sdf

def sdf_cylinder(center, radius, half_height, rotation_angles=(0.0, 0.0, 0.0), *, degrees=True):

    R = Rotation.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()
    R_T = R.T

    def _sdf(p):
        # ---- broadcast center with CSDL-safe ops ----
        if len(p.shape) == 1:
            c_b = center
        else:
            lead_shape   = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            c_b = csdl.tensordot(ones_leading, center, axes=None)

        # world -> local (cylinder frame: axis = local Y)
        p_local  = p - c_b
        last_ax  = len(p_local.shape) - 1
        pl       = csdl.tensordot(p_local, R_T, axes=([last_ax], [1]))  # (...,3)

        # ---- Avoid ellipsis-based slicing by projecting onto basis ----
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])

        px = csdl.tensordot(pl, ex, axes=([last_ax], [0]))  # shape: leading dims
        py = csdl.tensordot(pl, ey, axes=([last_ax], [0]))
        pz = csdl.tensordot(pl, ez, axes=([last_ax], [0]))

        eps   = 1e-12
        rxz_sq = px*px + pz*pz
        rxz    = csdl.sqrt(rxz_sq + eps)

        qx = rxz - radius
        qy = csdl.absolute(py) - half_height

        # Outside part: length(max(q,0))
        zeros   = np.broadcast_to(0.0, qx.shape)
        qx_pos  = csdl.maximum(qx, zeros)
        qy_pos  = csdl.maximum(qy, zeros)
        out_sq  = qx_pos*qx_pos + qy_pos*qy_pos
        outside = csdl.sqrt(out_sq + eps)

        # Inside part: min(max(qx,qy), 0)
        mx     = csdl.maximum(qx, qy)
        inside = csdl.minimum(mx, zeros)

        return outside + inside


    return _sdf

# ================= Numpy Versions for Explicit Op =================#


def sdf_box_np(center, half_size, rotation_angles=(0.0, 0.0, 0.0), *, degrees=True, order='xyz'):
    """
    Oriented box SDF (NumPy) using SciPy Rotation.
    center, half_size: (3,)
    rotation_angles: Euler angles in `order` (deg if degrees=True).
    """
    c = np.asarray(center, float).reshape(3)
    h = np.asarray(half_size, float).reshape(3)

    # SciPy rotation → 3x3 matrix; same semantics as your CSDL path
    Rm = Rotation.from_euler(order, rotation_angles, degrees=degrees).as_matrix()
    RT = Rm.T  # world -> box frame

    def _sdf(P):
        P = np.asarray(P, float)
        if P.ndim == 1:
            p_loc = (P - c) @ RT
            q = np.abs(p_loc) - h
            q_clip = np.maximum(q, 0.0)
            outside = np.linalg.norm(q_clip)
            inside  = np.minimum(np.max(q), 0.0)
            return outside + inside
        else:
            p_loc = (P - c[None, :]) @ RT
            q = np.abs(p_loc) - h[None, :]
            q_clip = np.maximum(q, 0.0)
            outside = np.linalg.norm(q_clip, axis=-1)
            inside  = np.minimum(np.max(q, axis=-1), 0.0)
            return outside + inside

    return _sdf


def sdf_sphere_np(center, radius):
    """
    Sphere SDF (NumPy).
    center: (3,), radius: float
    """
    c = np.asarray(center, float).reshape(3)
    r = float(radius)
    def _sdf(P):
        P = np.asarray(P, float)
        if P.ndim == 1:
            return np.linalg.norm(P - c) - r
        else:
            return np.linalg.norm(P - c[None, :], axis=-1) - r
    return _sdf

def sdf_plane_np(p0, normal):
    """
    Plane SDF (NumPy): signed distance; positive on the normal side.
    """
    p0 = np.asarray(p0, float).reshape(3)
    n  = np.asarray(normal, float).reshape(3)
    n  = n / (np.linalg.norm(n) + _EPS)

    def _sdf(P):
        P = np.asarray(P, float)
        if P.ndim == 1:
            return np.dot(P - p0, n)
        else:
            return (P - p0[None, :]) @ n
    return _sdf

def sdf_capsule_np(p1, p2, radius):
    """
    Capsule SDF (NumPy): segment [p1,p2] with radius.
    """
    a = np.asarray(p1, float).reshape(3)
    b = np.asarray(p2, float).reshape(3)
    r = float(radius)
    ab = b - a
    denom = np.dot(ab, ab) + _EPS

    def _sdf(P):
        P = np.asarray(P, float)
        if P.ndim == 1:
            t = np.clip(np.dot(P - a, ab) / denom, 0.0, 1.0)
            closest = a + t * ab
            return np.linalg.norm(P - closest) - r
        else:
            ap = P - a[None, :]
            t = np.clip((ap @ ab) / denom, 0.0, 1.0)           # (N,)
            closest = a[None, :] + t[:, None] * ab[None, :]    # (N,3)
            return np.linalg.norm(P - closest, axis=-1) - r
    return _sdf

def sdf_cylinder_np(center, radius, half_height, rotation_angles=(0.0, 0.0, 0.0), *, degrees=True):
    """
    NumPy version of the capped cylinder (IQ) oriented by Euler angles.
    """
    c = np.asarray(center, float).reshape(3)
    r = float(radius)
    h = float(half_height)
    RT = Rotation.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix().T

    def _sdf(P):
        P = np.asarray(P, float)
        if P.ndim == 1:
            pl = (P - c) @ RT
            rxz = np.linalg.norm([pl[0], pl[2]])
            qx = rxz - r
            qy = abs(pl[1]) - h
            outside = np.linalg.norm([max(qx, 0.0), max(qy, 0.0)])
            inside  = min(max(qx, qy), 0.0)
            return outside + inside
        else:
            pl = (P - c[None, :]) @ RT
            rxz = np.linalg.norm(pl[:, [0, 2]], axis=1)
            qx = rxz - r
            qy = np.abs(pl[:, 1]) - h
            outside = np.linalg.norm(np.c_[np.maximum(qx, 0.0), np.maximum(qy, 0.0)], axis=1)
            inside  = np.minimum(np.maximum(qx, qy), 0.0)
            return outside + inside

    return _sdf