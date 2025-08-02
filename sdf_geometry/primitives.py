import numpy as np

def sdf_box(p, bmin, bmax):
    bmin = np.array(bmin)
    bmax = np.array(bmax)
    center = 0.5 * (bmin + bmax)
    half_size = 0.5 * (bmax - bmin)
    q = np.abs(p - center) - half_size
    q_clip = np.maximum(q, 0.0)
    return np.linalg.norm(q_clip, axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)

def sdf_sphere(p, center, radius):
    return np.linalg.norm(p - center, axis=-1) - radius
 
