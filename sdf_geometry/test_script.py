import numpy as np
import csdl_alpha as csdl

from sdf_geometry.primitives import sdf_box, sdf_sphere
from sdf_geometry.operations import smooth_union, smooth_intersection, smooth_subtraction
from sdf_geometry.visualization import plot_2d_slice, plot_3d_isosurface

recorder = csdl.Recorder(inline = True)
recorder.start()

# ==== USER DEFINED PARAMETERS === 

# Choose Operation 1: Union, 2: Intersction, 3: Subtraction (A -B)
operation = 3   

# Create a 3D grid to evaluate SDF
res = 100
x = np.linspace(-5, 5, res)
y = np.linspace(-2, 2, res)
z = np.linspace(-2, 2, res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
P = csdl.Variable(value = np.stack([X, Y, Z], axis=-1))  # shape: (res, res, res, 3)

# Define shapes (sphere and box)
c = np.zeros(3)                          #Sphere center
R = 2.0                                  #Sphere radius
diag_lower = np.array([-4, -1, -1])      #Lower diagonal of box
diag_upper= np.array([4, 1, 1])          #Upper diagonal of box

# === END USER DEFINED PARAMETERS ===



# Evaluate box and sphere SDF
phi_box = sdf_box(P, bmin=diag_lower, bmax=diag_upper)
phi_sphere = sdf_sphere(P, center = c, radius = R)

# Compute Operation
if operation == 1:
    phi_union = smooth_union(phi_box, phi_sphere)
    plot_2d_slice(phi_union.value, x, y, title="2D Slice of Union SDF")
    plot_3d_isosurface(phi_union.value, x, y, z)
elif operation == 2:
    phi_intersection = smooth_intersection(phi_box, phi_sphere)
    plot_2d_slice(phi_intersection.value, x, y, title="2D Slice of Intersection SDF")
    plot_3d_isosurface(phi_intersection.value, x, y, z)
else:
    phi_subtraction = smooth_subtraction(phi_sphere, phi_box)
    plot_2d_slice(phi_subtraction.value, x, y, title="2D Slice of Subtraction SDF")
    plot_3d_isosurface(phi_subtraction.value, x, y, z)



