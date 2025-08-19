import numpy as np
import csdl_alpha as csdl
from prism import primitives as prim
from prism import operations as op
from prism.visualization import plot_2d_slice, plot_3d_isosurface
from prism.interference import collision_check

# --- Controls ---
SCENARIO = 1          # 1=apart, 2=touching, 3=overlap
GRID_N   = 64
PLOT     = True

# === Recorder ===
recorder = csdl.Recorder(inline=True)
recorder.start()

# === Grid helper ===
def make_grid(xmin=-2.0, xmax=2.0, N=64):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(xmin, xmax, N)
    z = np.linspace(xmin, xmax, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    P = csdl.Variable(value=np.stack([X, Y, Z], axis=-1))
    return x, y, z, P

# === Drone Geometry Parameters ===
arm_radius = 0.025     # arm capsule radius
rotor_radius = 0.35    # rotor disk radius

# Body (central box)
body_center = np.array([0.0, 0.0, -0.1])
body_half_size = np.array([0.2, 0.15, 0.15])
body_rotation = [0, 0, 0]  # deg

# Arms (capsules), hub at origin
hub = np.array([0.0, 0.0, 0.0])
arm_endpoints = [
    np.array([-0.5,  0.5, 0.0]),   # front-left
    np.array([ 0.5,  0.5, 0.0]),   # front-right
    np.array([ 0.5, -0.5, 0.0]),   # back-right
    np.array([-0.5, -0.5, 0.0]),   # back-left
]

# Rotors (disks = sphere ∩ two planes)
rotor_centers = [
    np.array([-0.5,  0.5, 0.15]),  # front-left
    np.array([-0.5, -0.5, 0.15]),  # back-left
    np.array([ 0.5,  0.5, 0.15]),  # front-right
    np.array([ 0.5, -0.5, 0.15]),  # back-right
]
rotor_bottoms = [
    np.array([-0.5,  0.5, 0.0]),
    np.array([-0.5, -0.5, 0.0]),
    np.array([ 0.5,  0.5, 0.0]),
    np.array([ 0.5, -0.5, 0.0]),
]

# === Build SDF primitives ===
# Body
phi_body = prim.sdf_box(body_center, body_half_size, body_rotation)

# Arms
phi_arms = [prim.sdf_capsule(hub, arm_end, arm_radius) for arm_end in arm_endpoints]

# Rotors (disks by intersection)
rotor_disks = []
for c, b in zip(rotor_centers, rotor_bottoms):
    sphere = prim.sdf_sphere(center=c, radius=rotor_radius)
    top_plane = prim.sdf_plane(p0=c, normal=np.array([0, 0,  1]))  # z >= c.z
    bottom_plane = prim.sdf_plane(p0=b, normal=np.array([0, 0, -1]))  # z <= b.z
    disk = op.intersection(sphere, top_plane, bottom_plane)  # sphere ∩ slab
    rotor_disks.append(disk)

# Composite drone SDF
phi_drone = op.union(phi_body, *phi_arms, *rotor_disks)

# === Obstacle sphere ===
r_ball = 0.30
# We'll reference the front-left rotor (index 0) so the scenarios are intuitive
c_rot_ref = rotor_centers[0]  # [-0.5, 0.5, 0.15]

if SCENARIO == 1:
    # Apart: place the ball left of the rotor rim with a gap
    # Rim along -x direction lies at x = c_rot_ref.x - rotor_radius
    c_ball_np = np.array([c_rot_ref[0] - (rotor_radius + r_ball + 0.2),
                          c_rot_ref[1],
                          c_rot_ref[2]])
elif SCENARIO == 2:
    # Just touching: tangent to the rotor rim along -x
    c_ball_np = np.array([c_rot_ref[0] - (rotor_radius + r_ball),
                          c_rot_ref[1],
                          c_rot_ref[2]])
else:
    # Overlap: penetrate by 1.5 along -x
    c_ball_np = np.array([c_rot_ref[0] - (rotor_radius + r_ball - 1.0),
                          c_rot_ref[1],
                          c_rot_ref[2]])

phi_ball = prim.sdf_sphere(center=csdl.Variable(value=c_ball_np), radius=r_ball)

# For visualization: union of drone and ball
phi_union = op.union(phi_drone, phi_ball)

# === Collision check (drone vs ball) ===
#x0_mid_np = 0.5 * (c_ball_np + c_rot_ref)       # Midpoint Seed
x0_mid_np = c_ball_np + 1E-1                    #Test gate offset
x0 = csdl.Variable(value=x0_mid_np)   

# Run
result = collision_check(phi_drone, phi_ball, x0, return_all=True)
x_star   = result[0].value
F_star   = result[1].value
a        = result[2].value
b        = result[3].value
pair_gap = result[4].value

print(f"Scenario = {SCENARIO}  (1 apart, 2 touching, 3 overlap)")
print("Stationary point x*:", x_star)
print("F(x*):", F_star)
print("Closest point on Drone (A):", a)
print("Closest point on Ball  (B):", b)
print("Pair gap distance:", pair_gap)

# === Visualize ===
xg, yg, zg, P = make_grid(-2.0, 2.0, GRID_N)
phi_val = phi_union(P).value  # evaluate on grid

if PLOT:
    plot_2d_slice(phi_val, xg, yg, title=f"Drone ∪ Ball (scenario={SCENARIO})")
    plot_3d_isosurface(phi_val, xg, yg, zg)
