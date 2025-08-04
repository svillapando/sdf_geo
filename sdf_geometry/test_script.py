import numpy as np
import csdl_alpha as csdl

from sdf_geometry.primitives import sdf_box, sdf_sphere, sdf_plane, sdf_capsule
from sdf_geometry.operations import smooth_union, smooth_intersection, smooth_subtraction
from sdf_geometry.visualization import plot_2d_slice, plot_3d_isosurface

recorder = csdl.Recorder(inline = True)
recorder.start()

# ==== USER DEFINED PARAMETERS === 

# Choose Operation 1: Union, 2: Intersction, 3: Subtraction (A -B)
operation = 1  

# Create a 3D grid to evaluate SDF
res = 100
x = np.linspace(-5, 5, res)
y = np.linspace(-5, 5, res)
z = np.linspace(-5, 5, res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
P = csdl.Variable(value = np.stack([X, Y, Z], axis=-1))  # shape: (res, res, res, 3)

# Define shapes

c = np.zeros(3)                             #Sphere center
R = 1.0                                     #Sphere radius

center = np.zeros(3)                        #Box Center
half_size = np.array([0.5, 1.5, 1.5])       #Box lengths in BODY axis
#rotation_angles = [30, 15, 60]             #Degrees about GLOBAL x, y, z axes
rotation_angles = [45, 0, 0]

p0 = np.zeros(3)                            #Point for plane
n = np.array([0,0,-1])                      #Normal vector 

p1 = np.array([-2, 0, -1])                   #Center of endcap 1 for cylinder
p2 = np.array([2, 0, 1])                    #Center of endcap 2 for cylinder

# === END USER DEFINED PARAMETERS ===



# Evaluate SDFs
phi_box = sdf_box(P, center, half_size, rotation_angles)
phi_sphere = sdf_sphere(P, c, R)
phi_plane = sdf_plane(P, p0, n)
phi_capsule = sdf_capsule(P, p1, p2, R)


# Compute Operations
if operation == 1:
    #phi_union = smooth_union(phi_box, phi_sphere)
    phi_union = smooth_union(phi_box, phi_capsule)
    plot_2d_slice(phi_union.value, x, y, title="2D Slice of Union SDF")
    plot_3d_isosurface(phi_union.value, x, y, z)
elif operation == 2:
    #phi_intersection = smooth_intersection(phi_box, phi_sphere)        #Box and Sphere
    phi_intersection = smooth_intersection(phi_sphere, phi_plane)       #Sphere and Plane (Hemisphere)
    plot_2d_slice(phi_intersection.value, x, y, title="2D Slice of Intersection SDF")
    plot_3d_isosurface(phi_intersection.value, x, y, z)
else:
    #phi_subtraction = smooth_subtraction(phi_box, phi_sphere)
    phi_subtraction = smooth_subtraction(phi_box, phi_capsule)
    plot_2d_slice(phi_subtraction.value, x, y, title="2D Slice of Subtraction SDF")
    plot_3d_isosurface(phi_subtraction.value, x, y, z)

#Test Individual point
point = np.array([2, 0, -1])
#sample = sdf_box(point, center, half_size, rotation_angles)
#sample = sdf_sphere(point, c, R)
#sample = sdf_plane(point, p0, n)
sample = sdf_capsule(point, p1, p2, R)
print('Phi value:', sample.value)