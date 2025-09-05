import numpy as np
import csdl_alpha as csdl
from prism import primitives as prim
from prism import operations as op
from prism.visualization import plot_2d_slice, plot_3d_isosurface, plot_isosurface_with_collision_points
from interference import collision_check, collision_check_kkt_LM

# --- Controls ---
SCENARIO = 1          # 1=apart, 2=touching, 3=overlap
GRID_N   = 128
PLOT     = True

# === Recorder ===
recorder = csdl.Recorder(inline=True)
recorder.start()

# === Grid helper ===
def make_grid(xmin=-3.0, xmax=3.0, N=64):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(xmin, xmax, N)
    z = np.linspace(xmin, xmax, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    P = csdl.Variable(value=np.stack([X, Y, Z], axis=-1))
    return x, y, z, P

# # === OLD Drone Geometry Parameters ===
# arm_radius = 0.025     # arm capsule radius
# rotor_radius = 0.35    # rotor disk radius

# # Body (central box)
# body_center = np.array([0.0, 0.0, -0.1])
# body_half_size = np.array([0.2, 0.15, 0.15])
# body_rotation = [0, 0, 0]  # deg

# # Arms (capsules), hub at origin
# hub = np.array([0.0, 0.0, 0.0])
# arm_endpoints = [
#     np.array([-0.5,  0.5, 0.0]),   # front-left
#     np.array([ 0.5,  0.5, 0.0]),   # front-right
#     np.array([ 0.5, -0.5, 0.0]),   # back-right
#     np.array([-0.5, -0.5, 0.0]),   # back-left
# ]

# # Rotors (disks = sphere ∩ two planes)
# rotor_centers = [
#     np.array([-0.5,  0.5, 0.15]),  # front-left
#     np.array([-0.5, -0.5, 0.15]),  # back-left
#     np.array([ 0.5,  0.5, 0.15]),  # front-right
#     np.array([ 0.5, -0.5, 0.15]),  # back-right
# ]
# rotor_bottoms = [
#     np.array([-0.5,  0.5, 0.0]),
#     np.array([-0.5, -0.5, 0.0]),
#     np.array([ 0.5,  0.5, 0.0]),
#     np.array([ 0.5, -0.5, 0.0]),
# ]


# === Drone Geometry Parameters (all dimensions in cm) ===
arm_radius = 0.75       # ~5 mm arm radius
rotor_radius = 6.3     # 5" prop ≈ 12.7 cm diameter -> 6.35 cm radius

# --- Body (central box) ---
body_center = np.array([0.0, 0.0, -1.5])     # slightly below hub
body_half_size = np.array([3.0, 3.0, 2.0])   # 6 x 6 x 4 cm body
body_rotation = [0, 0, 0]                    # Euler angles in degrees

# --- Arms (capsules) ---
hub = np.array([0.0, 0.0, 0.0])
arm_endpoints = [
    np.array([-8.0,  8.0, 0.0]),  # front-left
    np.array([ 8.0,  8.0, 0.0]),  # front-right
    np.array([ 8.0, -8.0, 0.0]),  # back-right
    np.array([-8.0, -8.0, 0.0]),  # back-left
]

# --- Rotors (disks = sphere ∩ planes) ---
# Centers slightly above the arm endpoints; thin disk thickness via bottom plane
rotor_centers = [
    np.array([-8.0,  8.0,  1.5]),  # front-left
    np.array([-8.0, -8.0,  1.5]),  # back-left
    np.array([ 8.0,  8.0,  1.5]),  # front-right
    np.array([ 8.0, -8.0,  1.5]),  # back-right
]

# Corresponding cutting planes (bottom of each disk)
rotor_bottoms = [
    np.array([-8.0,  8.0, 0.5]),
    np.array([-8.0, -8.0, 0.5]),
    np.array([ 8.0,  8.0, 0.5]),
    np.array([ 8.0, -8.0, 0.5]),
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
#r_ball = 0.30
r_ball = 3.0
c_rot_ref = rotor_centers[0]  

if SCENARIO == 1:
    # Apart: place the ball left of the rotor rim with a gap
    # Rim along -x direction lies at x = c_rot_ref.x - rotor_radius
    c_ball_np = np.array([c_rot_ref[0] - (rotor_radius + r_ball + 15.0),
                          c_rot_ref[1]- (rotor_radius - 5),
                          c_rot_ref[2]+ 15])
elif SCENARIO == 2:
    # Just touching: tangent to the rotor rim along -x
    c_ball_np = np.array([c_rot_ref[0] - (rotor_radius + r_ball),
                          c_rot_ref[1],
                          c_rot_ref[2]])
else:
    # Overlap: penetrate by 1.5 along -x
    c_ball_np = np.array([c_rot_ref[0] - (rotor_radius + r_ball - 0.2),
                          c_rot_ref[1],
                          c_rot_ref[2]])

phi_ball = prim.sdf_sphere(center=c_ball_np, radius=r_ball)

# For visualization: union of drone and ball
phi_union = op.union(phi_drone, phi_ball)

# === Collision check (drone vs ball) ===
x0_mid_np = 0.5 * (c_ball_np + c_rot_ref)       # Midpoint Seed
#x0_mid_np = c_ball_np + 3E-1                    #Test gate offset
x0 = csdl.Variable(value=x0_mid_np)   
eta_max = csdl.Variable(value = 0.3)

# Run
#result = collision_check(phi_drone, phi_ball, x0, eta_max, return_all=True)
result = collision_check_kkt_LM(phi_drone, phi_ball, x0, return_all=True)
# x_star   = result[0].value
# F_star   = result[1].value
# a        = result[2].value
# b        = result[3].value
# pair_gap = result[4].value

x_star   = result[0].value
a        = result[1].value
b        = result[2].value
lamA     = result[3].value
lamB     = result[4].value
#rho      = result[5].value
pair_gap = result[5].value


print(f"Scenario = {SCENARIO}  (1 apart, 2 touching, 3 overlap)")
print("Stationary point x*:", x_star)
#print("F(x*):", F_star)
print("Closest point on Drone (A):", a)
print("Closest point on Ball  (B):", b)
print("Pair gap distance:", pair_gap)

print("phi(drone) at p_drone:", phi_drone(a).value)
print("phi(env)   at p_env:",   phi_ball(b).value)

# === Visualize ===
xg, yg, zg, P = make_grid(-30.0, 30.0, GRID_N)
phi_val = phi_union(P).value  # evaluate on grid
test = csdl.Variable(shape=(3,), value=np.array([1.0, 0.0, 0.0]))  # pick a test point
phi_test = phi_drone(test)                                          # scalar
g        = csdl.reshape(csdl.derivative(ofs=phi_test, wrts=test), (3,))
eikonal  = csdl.norm(g)
print("‖∇phi‖ =", eikonal.value)
phi_env_test = phi_ball(test)                                          # scalar
g_env        = csdl.reshape(csdl.derivative(ofs=phi_env_test, wrts=test), (3,))
eikonal_env  = csdl.norm(g_env)
print("‖∇phi_env‖ =", eikonal_env.value)
if PLOT:
    #plot_2d_slice(phi_val, xg, yg, title=f"Drone ∪ Ball (scenario={SCENARIO})")
    #plot_3d_isosurface(phi_val, xg, yg, zg)
    plot_isosurface_with_collision_points(
        phi_val, xg, yg, zg,
        level=0.0,
        title=f"Drone ∪ Ball Collision Check (scenario={SCENARIO})",
        xstar=x_star,
        p_near_drone=a,
        p_near_env=b,
    )