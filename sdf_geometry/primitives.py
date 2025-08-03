import numpy as np
import csdl_alpha as csdl



def sdf_box(p, bmin, bmax):
    bmin = np.array(bmin)
    bmax = np.array(bmax)
    center = np.broadcast_to(0.5 * (bmin + bmax), p.shape)
    half_size = np.broadcast_to(0.5 * (bmax - bmin), p.shape)
    q = csdl.absolute(p - center) - half_size
    q_clip = csdl.maximum(q, np.broadcast_to(0.0, q.shape))
    last_axis_qclip = len(q_clip.shape) - 1
    last_axis_q = len(q.shape) -1
    return csdl.norm(q_clip,axes=(last_axis_qclip,)) + csdl.minimum(csdl.maximum(q, axes=(last_axis_q,)), np.broadcast_to(0.0, (100,100,100)))

def sdf_sphere(p, center, radius):
    center = np.broadcast_to(center, p.shape)
    dist = p - center
    last_axis_dist = len(dist.shape)-1
    return csdl.norm(dist, axes = (last_axis_dist,)) - radius
 
