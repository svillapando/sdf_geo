import numpy as np
import matplotlib.pyplot as plt

from sdf_geometry.primitives import sdf_box, sdf_sphere
from sdf_geometry.operations import smooth_union
from sdf_geometry.visualization import plot_2d_slice, plot_3d_isosurface


# Create a 3D grid to evaluate SDF
res = 100
x = np.linspace(-5, 5, res)
y = np.linspace(-2, 2, res)
z = np.linspace(-2, 2, res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
P = np.stack([X, Y, Z], axis=-1)  # shape: (res, res, res, 3)

# Define shapes (sphere and box)
c = np.zeros(3)                          #Sphere center
R = 2.0                                  #Sphere radius
diag_lower = np.array([-4, -1, -1])      #Lower diagonal of box
diag_upper= np.array([4, 1, 1])          #Upper diagonal of box


# Evaluate box and sphere SDF
phi_box = sdf_box(P, bmin=diag_lower, bmax=diag_upper)
phi_sphere = sdf_sphere(P, center = c, radius = R)

# Combine with smooth union
phi_union = smooth_union(phi_box, phi_sphere)


# Visualization
plot_2d_slice(phi_union, x, y, title="2D Slice of Union SDF")
plot_3d_isosurface(phi_union, x, y, z)

