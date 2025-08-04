import numpy as np
import csdl_alpha as csdl

from sdf_geometry.primitives import sdf_box, sdf_sphere, sdf_plane, sdf_capsule
from sdf_geometry.operations import smooth_union, smooth_intersection, smooth_subtraction
from sdf_geometry.visualization import plot_2d_slice, plot_3d_isosurface

recorder = csdl.Recorder(inline=True)
recorder.start()

# ==== USER DEFINED PARAMETERS === 

# Choose Operation 1: Union, 2: Intersection, 3: Subtraction (A - B)
operation = 1

# Create a 3D grid to evaluate SDF
res = 100
x = np.linspace(-5, 5, res)
y = np.linspace(-5, 5, res)
z = np.linspace(-5, 5, res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
P = csdl.Variable(value=np.stack([X, Y, Z], axis=-1))  # shape: (res, res, res, 3)

# Define shapes
c = np.zeros(3)                             # Sphere center
R = 1.0                                     # Sphere radius

center = np.zeros(3)                        # Box center
half_size = np.array([0.5, 1.5, 1.5])       # Box dimensions
rotation_angles = [45, 0, 0]                # Rotation in degrees

p0 = np.zeros(3)                            # Plane origin
n = np.array([0, 0, -1])                    # Plane normal

p1 = np.array([-2, 0, -1])                  # Capsule start point
p2 = np.array([2, 0, 1])                    # Capsule end point

# === END USER DEFINED PARAMETERS ===

# Build SDF functions
phi_box_fn = sdf_box(center, half_size, rotation_angles)
phi_sphere_fn = sdf_sphere(c, R)
phi_plane_fn = sdf_plane(p0, n)
phi_capsule_fn = sdf_capsule(p1, p2, R)


# Composite operations
if operation == 1:
    sdf_fn = smooth_union(phi_box_fn, phi_capsule_fn)
    phi = sdf_fn(P)
    plot_2d_slice(phi.value, x, y, title="2D Slice of Union SDF")
    plot_3d_isosurface(phi.value, x, y, z)

elif operation == 2:
    sdf_fn = smooth_intersection(phi_sphere_fn, phi_plane_fn)
    phi = sdf_fn(P)
    plot_2d_slice(phi.value, x, y, title="2D Slice of Intersection SDF")
    plot_3d_isosurface(phi.value, x, y, z)

else:
    sdf_fn = smooth_subtraction(phi_box_fn, phi_capsule_fn)
    phi = sdf_fn(P)
    plot_2d_slice(phi.value, x, y, title="2D Slice of Subtraction SDF")
    plot_3d_isosurface(phi.value, x, y, z)

# Test individual point
point = np.array([2, 0, -1])
sample = phi_capsule_fn(point)
print('Phi value:', sample.value)
