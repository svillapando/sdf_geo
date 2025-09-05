import numpy as np
import csdl_alpha as csdl
from scipy.spatial.transform import Rotation

def sdf_box(center, half_size, rotation_angles, degrees=True):
    def _sdf(p):
        R = Rotation.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()
        R_T = R.T
        eps = 1e-12

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
            # ||q_clip||_2 with epsilon (safe)
            norm_sq = csdl.vdot(q_clip, q_clip)
            norm_val = csdl.sqrt(norm_sq + eps)
        else:
            # sum over last axis
            norm_sq = csdl.sum(q_clip * q_clip, axes=(len(q_clip.shape) - 1,))
            norm_val = csdl.sqrt(norm_sq + eps)

        q_max = csdl.maximum(q, axes=(len(q.shape) - 1,))
        correction = csdl.minimum(q_max, np.broadcast_to(0.0, q_max.shape))

        return norm_val + correction

    return _sdf


def sdf_sphere(center, radius):
    def _sdf(p):
        eps = 1e-12

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
            return csdl.sqrt(norm_sq + eps) - radius
        else:
            norm_sq = csdl.sum(dist * dist, axes=(last_axis,))
            return csdl.sqrt(norm_sq + eps) - radius

    return _sdf


def sdf_plane(p0, normal):
    def _sdf(p):
        eps = 1e-12
        # ---- normalize normal (true signed distance scale) ----
        n_norm = csdl.sqrt(csdl.vdot(normal, normal) + eps)
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
        eps = 1e-12

        # ---- broadcast p1 with CSDL-safe ops ----
        if len(p.shape) == 1:
            p1_b = p1
        else:
            lead_shape   = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            p1_b = csdl.tensordot(ones_leading, p1, axes=None)

        ba = p2 - p1                          # (3,)
        axis_len_sq = csdl.sum(ba * ba) + eps # add eps before division

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
            d = csdl.sqrt(dist_sq + eps) - radius
        else:
            diff = p - proj
            dist_sq = csdl.sum(diff * diff, axes=(len(p.shape) - 1,))
            d = csdl.sqrt(dist_sq + eps) - radius
        return d

    return _sdf
