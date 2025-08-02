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

def smooth_union(a, b, k=8.0):
    return -np.log(np.exp(-k*a) + np.exp(-k*b)) / k


# Create a 3D grid
res = 100
x = np.linspace(-5, 5, res)
y = np.linspace(-2, 2, res)
z = np.linspace(-2, 2, res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
P = np.stack([X, Y, Z], axis=-1)  # shape: (res, res, res, 3)

# Evaluate each SDF
phi_box = sdf_box(P, bmin=np.array([-4, -1, -1]), bmax=np.array([4, 1, 1]))

# Define sphere geomemtry
c = np.zeros(3)
R = 2.0
phi_sphere = sdf_sphere(P, center = c, radius = R)

# Combine with smooth union
phi_union = smooth_union(phi_box, phi_sphere)




import matplotlib.pyplot as plt

# Take a Z slice (middle of volume)
z_idx = res // 2
phi_slice = phi_union[:, :, z_idx]

plt.figure()
cf = plt.contourf(x, y, phi_slice.T, levels=50, cmap='RdBu')

plt.contour(x, y, phi_slice.T, levels=[0.0], colors='black', linewidths=2)
plt.title("φ(x, y, z=0) — Union of Box and Sphere")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cf, label='Signed Distance')
plt.axis('equal')
plt.grid(True)
plt.show()

import pyvista as pv


# 1. Convert 3D numpy grid into a PyVista UniformGrid
grid = pv.ImageData()
grid.dimensions = np.array(phi_union.shape) 
grid.spacing = (
    x[1] - x[0],
    y[1] - y[0],
    z[1] - z[0],
)
grid.origin = (x[0], y[0], z[0])  # bottom corner
grid.point_data["phi"] = phi_union.flatten(order="F")  # Fortran order required

# 2. Extract the isosurface (φ = 0)
contour = grid.contour(isosurfaces=[0.0], scalars="phi")

# 3. Visualize
plotter = pv.Plotter()
plotter.add_mesh(contour, color="blue", opacity=0.7)
plotter.add_axes()
plotter.add_title("Isosurface of φ = 0")
plotter.show()




#--- Testing SDF Poitns ---
test_pts = np.array([
    [0, 0, 0],        # Inside both
    [3.9, 0, 0],      # Near box edge
    [0, 0, 2.1],      # Just outside sphere
    [-5, 0, 0]        # Outside both
])

for pt in test_pts:
    val = smooth_union(sdf_box(pt, [-4, -1, -1], [4, 1, 1]), sdf_sphere(pt))
    print(f"φ({pt}) = {val:.4f}")
