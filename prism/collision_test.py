import numpy as np
import csdl_alpha as csdl
from prism import primitives as prim
from prism import operations as op
from prism.visualization import plot_2d_slice, plot_3d_isosurface
from prism.interference import collision_check
# SCENARIO: 1=apart, 2=touching, 3=overlap
SCENARIO = 3
GRID_N   = 64
PLOT  = True

recorder = csdl.Recorder(inline=True)
recorder.start()

def make_grid(xmin=-3, xmax=3, N=64):
    x = np.linspace(xmin, xmax, N); y = np.linspace(xmin, xmax, N); z = np.linspace(xmin, xmax, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, csdl.Variable(value=np.stack([X, Y, Z], axis=-1))

# centers/radii
r1 = 1.0; r2 = 1.0
if SCENARIO == 1:
    c1_np, c2_np = np.array([-1.5, 0.0, 0.0]), np.array([+1.5, 0.0, 0.0])
elif SCENARIO == 2:
    c1_np, c2_np = np.array([-1.0, 0.0, 0.0]), np.array([+1.0, 0.0, 0.0])
else:
    c1_np, c2_np = np.array([-0.8, 0.0, 0.0]), np.array([+0.8, 0.0, 0.0])

c1 = csdl.Variable(value=c1_np)
c2 = csdl.Variable(value=c2_np)
phi1 = prim.sdf_sphere(c1, r1)
phi2 = prim.sdf_sphere(c2, r2)
phi_union = op.union(phi1, phi2)
x0 = (0.5)*(c1_np + c2_np)      # Initial guess at the midpoint
x0 = csdl.Variable(value  = x0) # Test both NumPy and CSDL Variable
result = collision_check(phi1, phi2, x0, return_all=True)
x_star = result[0].value
F_star = result[1].value
a = result[2].value
b = result[3].value
pair_gap = result[4].value
# Access results
print("Stationary point x*:", x_star)
print("F(x*):", F_star)
print("Closest point on A:", a)
print("Closest point on B:", b)
print("Pair gap distance:", pair_gap)

# visualize
xg, yg, zg, P = make_grid(-3, 3, GRID_N)
phi = phi_union(P)

if PLOT:
    plot_2d_slice(phi.value, xg, yg, title=f"Spheres (scenario={SCENARIO})")
    plot_3d_isosurface(phi.value, xg, yg, zg)
