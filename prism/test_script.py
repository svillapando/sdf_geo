import numpy as np
import csdl_alpha as csdl

import primitives as prim
import operations as op
from visualization import plot_2d_slice, plot_3d_isosurface

recorder = csdl.Recorder(inline=True)
recorder.start()

# ==== USER DEFINED PARAMETERS === 

# Choose Operation 1: Union, 2: Intersection, 3: Subtraction (A - B)
operation = 3

# Create a 3D grid to evaluate SDF
res = 100
x = np.linspace(-2, 2, res)
y = np.linspace(-2, 2, res)
z = np.linspace(-2, 2, res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
P = csdl.Variable(value=np.stack([X, Y, Z], axis=-1))  # shape: (res, res, res, 3)

# Define shapes
c = csdl.Variable(value = np.zeros(3))                             # Sphere center
R = csdl.Variable(value = 1.0)                                     # Sphere radius

center = csdl.Variable(value = np.zeros(3))                        # Box Center
half_size = csdl.Variable(value = np.array([0.25, 1.5, 1.5]))      # Box dimensions
rotation_angles = [0, 0, 0]                                        # Rotation in degrees

p0 = np.zeros(3)                            # Plane origin
n = np.array([0, 0, -1])                    # Plane normal


p1 = csdl.Variable(value = np.array([-1, 0, 0]))                # Capsule start point
p2 = csdl.Variable(value = np.array([1, 0, 0]))                 # Capsule end point

# Test single point
point = csdl.Variable(value=np.array([2.0, 0.0, -1.0]))

# === END USER DEFINED PARAMETERS ===

# Build SDF functions
phi_box_fn = prim.sdf_box(center, half_size, rotation_angles)
phi_sphere_fn = prim.sdf_sphere(c, R)
phi_plane_fn = prim.sdf_plane(p0, n)
phi_capsule_fn = prim.sdf_capsule(p1, p2, R)


# Helper: evaluate, plot, and sample point
def _run_case(title, sdf_fn):
    phi = sdf_fn(P)
    plot_2d_slice(phi.value, x, y, title=title)
    plot_3d_isosurface(phi.value, x, y, z)
    print(f'{title} @ point:', sdf_fn(point).value)


if operation == 1:
    # UNION: box ∪ sphere ∪ capsule
    sdf_fn = op.union(phi_box_fn, phi_sphere_fn)
    _run_case("Union", sdf_fn)

elif operation == 2:
    # INTERSECTION: sphere ∩ slab (two parallel planes)
    h = 0.4
    plane_up = prim.sdf_plane(p0 + h * n,  n)   # keep side n·(x - (p0+h n)) >= 0  => z ≥  h (if n=[0,0,1])
    plane_dn = prim.sdf_plane(p0 - h * n, -n)   # keep side (-n)·(x - (p0-h n)) ≥ 0 => z ≤ -h (mirrored)
    # Note: with n = [0,0,-1], these two keep -h ≤ z ≤ h (a slab).
    sdf_fn = op.intersection(phi_sphere_fn, plane_up, plane_dn)
    _run_case("Intersection", sdf_fn)

else:
    # SUBTRACTION: box - sphere - plane (half arch)
    sdf_fn = op.subtraction(phi_box_fn, phi_sphere_fn, phi_plane_fn)
    _run_case("Subtraction", sdf_fn)

