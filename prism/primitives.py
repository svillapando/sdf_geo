import numpy as np
import csdl_alpha as csdl
from scipy.spatial.transform import Rotation

# test
def sdf_box(center, half_size, rotation_angles, degrees=True):
    def _sdf(p):
        R = Rotation.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()
        R_T = R.T

        center_bcast = np.broadcast_to(center, p.shape)
        p_local = p - center_bcast  # (..., 3)

        last_axis = len(p_local.shape) - 1
        p_rot = csdl.tensordot(p_local, R_T, axes=([last_axis], [1]))  # (..., 3)

        half_bcast = np.broadcast_to(half_size, p.shape)
        q = csdl.absolute(p_rot) - half_bcast

        q_clip = csdl.maximum(q, np.broadcast_to(0.0, q.shape))

        if len(q_clip.shape) == 1:
            norm_val = csdl.norm(q_clip)
        else:
            norm_val = csdl.norm(q_clip, axes=(len(q.shape) - 1,))

        q_max = csdl.maximum(q, axes=(len(q.shape) - 1,))
        correction = csdl.minimum(q_max, np.broadcast_to(0.0, q_max.shape))

        return norm_val + correction

    return _sdf


def sdf_sphere(center, radius):
    def _sdf(p):
        center_bcast = np.broadcast_to(center, p.shape)
        dist = p - center_bcast
        last_axis = len(p.shape) - 1

        if len(p.shape) == 1:
            return csdl.norm(dist) - radius
        else:
            return csdl.norm(dist, axes=(last_axis,)) - radius

    return _sdf


def sdf_plane(p0, normal):
    def _sdf(p):
        p0_bcast = np.broadcast_to(p0, p.shape)
        n_bcast = np.broadcast_to(normal, p.shape)
        delta = p - p0_bcast
        return csdl.sum(delta * n_bcast, axes=(len(p.shape) - 1,))
    return _sdf


def sdf_capsule(p1, p2, radius):
    def _sdf(p):
        ba = p2 - p1  # (3,)
        axis_len_sq = np.sum(ba * ba)

        pa = p - np.broadcast_to(p1, p.shape)  # (..., 3)

        t_numer = csdl.sum(pa * np.broadcast_to(ba, p.shape), axes=(len(p.shape) - 1,))
        t = t_numer / axis_len_sq
        t_clamped = csdl.maximum(np.broadcast_to(0, t.shape),
                                 csdl.minimum(np.broadcast_to(1, t.shape), t))

        t_clamped_expanded = csdl.expand(t_clamped, out_shape=p.shape, action='ijk->ijkv')
        proj = np.broadcast_to(p1, p.shape) + t_clamped_expanded * np.broadcast_to(ba, p.shape)

        if len(p.shape) == 1:
            d = csdl.norm(p - proj) - radius
        else:
            d = csdl.norm(p - proj, axes=(len(p.shape) - 1,)) - radius
        return d

    return _sdf
