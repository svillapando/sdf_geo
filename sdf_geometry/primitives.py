import numpy as np
import csdl_alpha as csdl
from scipy.spatial.transform import Rotation


def sdf_box(p, center, half_size, rotation_angles, degrees=True):

    # Compute rotation matrix from global x, y, z axes
    R = Rotation.from_euler('xyz', rotation_angles, degrees=degrees).as_matrix()
    R_T = R.T  

    center_bcast = np.broadcast_to(center, p.shape)
    p_local = p - center_bcast  # (..., 3)
 
    last_axis = len(p_local.shape) - 1
    p_rot = csdl.tensordot(p_local, R_T, axes=([last_axis], [1]))  # (..., 3)

    half_bcast = np.broadcast_to(half_size, p.shape)
    q = csdl.absolute(p_rot) - half_bcast

    # Compute distance to surface (only if outside)
    q_clip = csdl.maximum(q, np.broadcast_to(0.0, q.shape))
    norm_val = csdl.norm(q_clip, axes=(len(q.shape)-1,))

    # Compute distance to surface (only if inside)
    q_max = csdl.maximum(q, axes=(len(q.shape)-1,))
    correction = csdl.minimum(q_max, np.broadcast_to(0.0, q_max.shape))

    # If inside, norm_val -> 0, correction -> negative. If outside, norm_val -> positive, correction -> 0 
    return norm_val + correction

def sdf_sphere(p, center, radius):
    center = np.broadcast_to(center, p.shape)
    dist = p - center
    last_axis_dist = len(dist.shape)-1
    return csdl.norm(dist, axes = (last_axis_dist,)) - radius
 
