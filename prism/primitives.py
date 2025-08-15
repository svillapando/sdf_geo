import numpy as np
import csdl_alpha as csdl
from scipy.spatial.transform import Rotation


def sdf_box(center, half_size, rotation_angles, degrees=True):
    def _sdf(p):
        R = Rotation.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()
        R_T = R.T


        if len(p.shape) == 1:
            # Single point: no broadcast needed
            center_bcast = center
        else:
            # General case: outer product with ones over leading dims
            lead_shape = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            center_bcast = csdl.tensordot(ones_leading, center, axes=None)


        p_local = p - center_bcast  # (..., 3)

        last_axis = len(p_local.shape) - 1
        p_rot = csdl.tensordot(p_local, R_T, axes=([last_axis], [1]))  # (..., 3)

  
        if len(p.shape) == 1:
            # Single point: no broadcast needed
            half_bcast = half_size
        else:
            # General case: outer product with ones over leading dims
            lead_shape   = p.shape[:-1]                              # leading dims of p (could be ())
            ones_leading = np.ones(lead_shape)
            half_bcast = csdl.tensordot(ones_leading, half_size, axes=None)

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
     
        if len(p.shape) == 1:
            # Single point: no broadcast needed
            center_bcast = center
        else:
            # General case: outer product with ones over leading dims
            lead_shape = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            center_bcast = csdl.tensordot(ones_leading, center, axes=None)

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
        if len(p.shape) == 1:
            # Single point: no broadcast needed
            p1_bcast = p1
        else:
            # General case: outer product with ones over leading dims
            lead_shape = p.shape[:-1]
            ones_leading = np.ones(lead_shape)
            p1_bcast = csdl.tensordot(ones_leading, p1, axes=None)

        ba = p2-p1 # (3,) 
        axis_len_sq = csdl.sum(ba * ba)

        pa = p - p1_bcast  # (..., 3)

        last = len(p.shape) - 1
        # instead of: t_numer = csdl.sum(pa * ba, axes=(last,))
        t_numer = csdl.tensordot(pa, ba, axes=([last], [0]))   # shape (...)

        t = t_numer / axis_len_sq
        # clamp with floats (avoid int-casting t)
        t = csdl.minimum(csdl.maximum(t, np.broadcast_to(0.0, t.shape)),
                        np.broadcast_to(1.0, t.shape))

        # instead of making t_vec then multiplying by ba (which would broadcast),
        # directly make the outer product (…,1)·(3,) → (…,3)
        proj = p1_bcast + csdl.reshape(csdl.tensordot(t, ba, axes=None), p.shape) 

        if len(p.shape) == 1:
            d = csdl.norm(p - proj) - radius
        else:
            d = csdl.norm(p - proj, axes=(len(p.shape) - 1,)) - radius
        return d

    return _sdf
