
from typing import Callable, Dict, Tuple, List, Optional, Union
import numpy as np
import csdl_alpha as csdl

# class BroadphaseOp(csdl.CustomExplicitOperation):
#     def __init__(self, A_np_fn, B_np_fn=None, *, K=8, max_refine=8,
#                  enable_phase2=False, expects_A_params=False):
#         super().__init__()
#         self.A_np_fn = A_np_fn              # e.g. drone_phi_np
#         self.B_np_fn = B_np_fn              # optional numeric φ_B
#         self.K = int(K); self.max_refine = int(max_refine)
#         self.enable_phase2 = bool(enable_phase2 and (B_np_fn is not None))
#         self.expects_A_params = bool(expects_A_params)

#     def evaluate(self, inputs):
#         self.declare_input('pos_w_m',   inputs.pos_w_m)
#         self.declare_input('quat_wxyz', inputs.quat_wxyz)

#         # Only declare geometry inputs if you want the op to read DVs for A:
#         if self.expects_A_params:
#             self.declare_input('body_center_cm',   inputs.body_center_cm)
#             self.declare_input('body_half_cm',     inputs.body_half_cm)
#             self.declare_input('body_angles_deg',  inputs.body_angles_deg)
#             self.declare_input('arm_radius_cm',    inputs.arm_radius_cm)
#             self.declare_input('arm_ends_cm',      inputs.arm_ends_cm)       # (N_arm,3)
#             self.declare_input('rotor_centers_cm', inputs.rotor_centers_cm)  # (N_rot,3)
#             self.declare_input('rotor_bottoms_cm', inputs.rotor_bottoms_cm)  # (N_rot,3)
#             self.declare_input('rotor_radii_cm',   inputs.rotor_radii_cm)    # (N_rot,)

#         C_b_cm = self.create_output('C_b_cm', (self.K, 3))
#         r_cell = self.create_output('r_cell', (1,))
#         return csdl.VariableGroup(C_b_cm=C_b_cm, r_cell=r_cell)
    
#     # --- pure NumPy compute ---
#     @staticmethod
#     def _quat_to_R_wb(q):  # WORLD <- BODY (w,x,y,z)
#         w,x,y,z = q; n = np.linalg.norm(q); 
#         if n == 0: return np.eye(3)
#         w,x,y,z = q/n
#         return np.array([
#             [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
#             [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
#             [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
#         ], float)

#     def compute(self, inp, out):
#         # pose (numbers)
#         p_w  = np.asarray(inp['pos_w_m'],   float).reshape(3)
#         q    = np.asarray(inp['quat_wxyz'], float).reshape(4)
#         R_wb = self._quat_to_R_wb(q); R_bw = R_wb.T

#         # Build φ_A in WORLD meters (calls your drone_phi_np)
#         if self.expects_A_params:
#             A = dict(
#                 body_center   = np.asarray(inp['body_center_cm'],   float),
#                 body_half     = np.asarray(inp['body_half_cm'],     float),
#                 body_angles   = np.asarray(inp['body_angles_deg'],  float),
#                 arm_radius    = float(np.asarray(inp['arm_radius_cm'], float)),
#                 arm_endpoints = np.asarray(inp['arm_ends_cm'],      float).reshape(-1,3).tolist(),
#                 rotor_centers = np.asarray(inp['rotor_centers_cm'], float).reshape(-1,3).tolist(),
#                 rotor_bottoms = np.asarray(inp['rotor_bottoms_cm'], float).reshape(-1,3).tolist(),
#                 rotor_radius  = np.asarray(inp['rotor_radii_cm'],   float).reshape(-1),
#             )
#             def phi_A_body_cm_np(P_b_cm):
#                 return self.A_np_fn(P_b_cm, params=A)   # ← with params
#         else:
#             def phi_A_body_cm_np(P_b_cm):
#                 return self.A_np_fn(P_b_cm)            # ← no params

#         def phi_A_world_m_np(P_w_m):
#             Pw = np.asarray(P_w_m, float).reshape(-1,3)
#             Pb_m  = (Pw - p_w) @ R_wb.T
#             Pb_cm = 100.0 * Pb_m
#             return 0.01 * phi_A_body_cm_np(Pb_cm)
        
#         phi_B_world_m_np = self.B_np_fn  # may be None

#         # --- autosize root bbox around A by probing φ_A in WORLD m ---
#         def autosize_root_bbox():
#             dirs = []
#             s = 1/np.sqrt(2); t = 1/np.sqrt(3)
#             axes  = [[+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]]
#             edges = [[+s,+s,0],[+s,-s,0],[-s,+s,0],[-s,-s,0],[+s,0,+s],[+s,0,-s],[-s,0,+s],[-s,0,-s],
#                      [0,+s,+s],[0,+s,-s],[0,-s,+s],[0,-s,-s]]
#             corners = [[+t,+t,+t],[+t,+t,-t],[+t,-t,+t],[+t,-t,-t],[-t,+t,+t],[-t,+t,-t],[-t,-t,+t],[-t,-t,-t]]
#             for d in axes+edges+corners:
#                 v = np.asarray(d,float); v /= np.linalg.norm(v); dirs.append(v)
#             radii = np.linspace(0.05, 0.80, 32)  # m
#             max_hit, margin = 0.25, 0.05
#             for d in dirs:
#                 P = p_w[None,:] + radii[:,None]*d[None,:]
#                 ph = phi_A_world_m_np(P)
#                 idx = np.argmax(ph > 0.01) if (ph > 0.01).any() else (len(radii)-1)
#                 hit_r = radii[min(idx+1, len(radii)-1)]
#                 max_hit = max(max_hit, float(hit_r))
#             half = max_hit + margin
#             return (p_w[0]-half,p_w[0]+half,p_w[1]-half,p_w[1]+half,p_w[2]-half,p_w[2]+half)

#         # voxel structure (NumPy-only)
#         class Vox:
#             def __init__(self, nodes, L, b):
#                 self.nodes=np.asarray(nodes,np.int64).reshape(-1,3); self.L=int(L)
#                 self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax = map(float,b)
#             @classmethod
#             def root(cls,b): return cls(np.array([[0,0,0]],np.int64),0,b)
#             def _sizes(self):
#                 W=self.xmax-self.xmin; H=self.ymax-self.ymin; D=self.zmax-self.zmin; n=1<<self.L
#                 dx,dy,dz=W/n,H/n,D/n; return dx,dy,dz,0.5*dx,0.5*dy,0.5*dz
#             def centers(self):
#                 dx,dy,dz,_,_,_=self._sizes(); ijk=self.nodes.astype(float)
#                 cx=self.xmin+(ijk[:,0]+0.5)*dx; cy=self.ymin+(ijk[:,1]+0.5)*dy; cz=self.zmin+(ijk[:,2]+0.5)*dz
#                 return np.column_stack([cx,cy,cz]).astype(np.float32,copy=False)
#             def half_sizes(self): _,_,_,hx,hy,hz=self._sizes(); return float(hx),float(hy),float(hz)
#             def radius(self): hx,hy,hz=self.half_sizes(); return float(np.linalg.norm([hx,hy,hz]))
#             def subdivide(self):
#                 i,j,k=self.nodes[:,0],self.nodes[:,1],self.nodes[:,2]
#                 kids=np.array([np.stack(((i<<1)|a,(j<<1)|b,(k<<1)|c),1)
#                                for a in (0,1) for b in (0,1) for c in (0,1)]).reshape(-1,3)
#                 return Vox(kids,self.L+1,(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax))
#             def restrict(self,mask): self.nodes=self.nodes[np.asarray(mask,bool)]

#         # refine
#         vox = Vox.root(autosize_root_bbox())
#         for _ in range(self.max_refine):
#             vox = vox.subdivide()
#             r   = vox.radius()
#             C   = vox.centers()
#             keep1 = (phi_A_world_m_np(C) <= r)     # Phase-1 using A
#             vox.restrict(keep1)
#             if self.enable_phase2 and len(vox.nodes)>0:
#                 C2 = vox.centers()
#                 phiB = phi_B_world_m_np(C2)
#                 min_u = np.min(phiB) + r
#                 keep2 = (phiB <= (min_u + r))
#                 vox.restrict(keep2)

#         C_final = vox.centers()
#         hx,hy,hz = vox.half_sizes()
#         r_cell = float(np.linalg.norm([hx,hy,hz]))  # m

#         if len(C_final)==0:
#             C_top = np.zeros((self.K,3), float)
#         else:
#             # rank: use φ_B if available, else φ_A
#             scores = (phi_B_world_m_np(C_final) if self.enable_phase2 else phi_A_world_m_np(C_final)) - r_cell
#             order  = np.argsort(scores); k = min(self.K, len(order))
#             C_top  = C_final[order[:k]]
#             if k < self.K:
#                 pad = C_top[-1:] if k>0 else np.zeros((1,3),float)
#                 C_top = np.vstack([C_top, np.repeat(pad, self.K-k, axis=0)])

#         # return BODY-frame cm candidates (constants for narrow-phase)
#         C_b_m  = (C_top - p_w[None,:]) @ R_bw.T
#         out['C_b_cm'] = (100.0 * C_b_m).astype(float)   # (K,3), cm
#         out['r_cell'] = np.array([r_cell], float)       # (1,), m

#     def compute_derivatives(self, *_): pass  # non-smooth selection


# def collision_check(
#     phi_A_body_cm_np,           # your drone_sdf_np (NumPy; BODY cm)
#     phi_B_world_m,              # CSDL evaluator (WORLD m)
#     pos_w_m, quat_wxyz,         # CSDL vars
#     *, K=5, max_refine=8, enable_phase2=False, phi_B_world_m_np=None
# ):
#     # Feed pose into the op; (no A params here since your drone_sdf_np is constant for now)
#     inputs = csdl.VariableGroup()
#     inputs.pos_w_m, inputs.quat_wxyz = pos_w_m, quat_wxyz

#     op = BroadphaseOp(
#     A_np_fn=phi_A_body_cm_np,
#     B_np_fn=phi_B_world_m_np,
#     K=K, max_refine=max_refine, enable_phase2=enable_phase2
#     )
#     outs = op.evaluate(inputs)
#     C_b_cm, r_cell = outs.C_b_cm, outs.r_cell[0]   # (K,3), scalar

#     # Narrow phase (differentiable wrt pose; same as you had)
#     def quat_to_R(q):
#         qw,qx,qy,qz = q[0],q[1],q[2],q[3]
#         r00 = 1-2*(qy*qy+qz*qz); r01 = 2*(qx*qy-qw*qz); r02 = 2*(qx*qz+qw*qy)
#         r10 = 2*(qx*qy+qw*qz);   r11 = 1-2*(qx*qx+qz*qz); r12 = 2*(qy*qz-qw*qx)
#         r20 = 2*(qx*qz-qw*qy);   r21 = 2*(qy*qz+qw*qx);   r22 = 1-2*(qx*qx+qy*qy)
#         return csdl.reshape(csdl.vstack((r00,r01,r02,r10,r11,r12,r20,r21,r22)), (3,3))

#     C_b_m  = C_b_cm / 100.0
#     R_wb   = quat_to_R(quat_wxyz)
#     C_rot  = csdl.matmat(C_b_m, csdl.transpose(R_wb))
#     Kk     = C_b_m.shape[0]
#     pos_t  = csdl.expand(pos_w_m, out_shape=(Kk,3), action='i->ji')
#     C_w    = pos_t + C_rot                           # (K,3) WORLD m

#     phiB_K = phi_B_world_m(C_w)                      # (K,)
#     d_soft = csdl.minimum(phiB_K - r_cell)           # conservative soft-min
#     return d_soft





from typing import Callable, Dict, Tuple, List, Optional, Union
import numpy as np
import csdl_alpha as csdl

# class BroadphaseOp(csdl.CustomExplicitOperation):
#     def __init__(self, A_np_fn, B_np_fn=None, *, K=8, max_refine=8,
#                  enable_phase2=False, expects_A_params=False, K_pool=None):
#         super().__init__()
#         self.A_np_fn = A_np_fn              # NumPy φ_A in BODY cm
#         self.B_np_fn = B_np_fn              # NumPy φ_B in WORLD m (optional)
#         self.K = int(K)
#         self.K_pool = int(K_pool) if K_pool is not None else max(3*int(K), int(K))
#         self.max_refine = int(max_refine)
#         self.enable_phase2 = bool(enable_phase2 and (B_np_fn is not None))
#         self.expects_A_params = bool(expects_A_params)

#         # Stability caches
#         self._root_bounds = None            # frozen voxel root bbox after first autosize
#         self._cache = None                  # {'C_b_cm':(K,3), 'r_cell':(1,)}

#     def evaluate(self, inputs):
#         self.declare_input('pos_w_m',   inputs.pos_w_m)
#         self.declare_input('quat_wxyz', inputs.quat_wxyz)

#         if self.expects_A_params:
#             self.declare_input('body_center_cm',   inputs.body_center_cm)
#             self.declare_input('body_half_cm',     inputs.body_half_cm)
#             self.declare_input('body_angles_deg',  inputs.body_angles_deg)
#             self.declare_input('arm_radius_cm',    inputs.arm_radius_cm)
#             self.declare_input('arm_ends_cm',      inputs.arm_ends_cm)       # (N_arm,3)
#             self.declare_input('rotor_centers_cm', inputs.rotor_centers_cm)  # (N_rot,3)
#             self.declare_input('rotor_bottoms_cm', inputs.rotor_bottoms_cm)  # (N_rot,3)
#             self.declare_input('rotor_radii_cm',   inputs.rotor_radii_cm)    # (N_rot,)

#         C_b_cm = self.create_output('C_b_cm', (self.K, 3))
#         r_cell = self.create_output('r_cell', (1,))

#         outs = csdl.VariableGroup()
#         outs.C_b_cm = C_b_cm
#         outs.r_cell = r_cell
#         return outs

#     @staticmethod
#     def _quat_to_R_wb(q):  # WORLD <- BODY (w,x,y,z)
#         w, x, y, z = q
#         n = np.linalg.norm(q)
#         if n == 0.0:
#             return np.eye(3)
#         w, x, y, z = q / n
#         return np.array([
#             [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
#             [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
#             [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
#         ], float)

#     def compute(self, inp, out):
#         # Pose (numbers)
#         p_w  = np.asarray(inp['pos_w_m'],   float).reshape(3)
#         q    = np.asarray(inp['quat_wxyz'], float).reshape(4)
#         R_wb = self._quat_to_R_wb(q)
#         R_bw = R_wb.T

#         # Build φ_A in WORLD meters via your NumPy BODY-cm callable
#         if self.expects_A_params:
#             A = dict(
#                 body_center   = np.asarray(inp['body_center_cm'],   float),
#                 body_half     = np.asarray(inp['body_half_cm'],     float),
#                 body_angles   = np.asarray(inp['body_angles_deg'],  float),
#                 arm_radius    = float(np.asarray(inp['arm_radius_cm'], float)),
#                 arm_endpoints = np.asarray(inp['arm_ends_cm'],      float).reshape(-1, 3).tolist(),
#                 rotor_centers = np.asarray(inp['rotor_centers_cm'], float).reshape(-1, 3).tolist(),
#                 rotor_bottoms = np.asarray(inp['rotor_bottoms_cm'], float).reshape(-1, 3).tolist(),
#                 rotor_radius  = np.asarray(inp['rotor_radii_cm'],   float).reshape(-1),
#             )
#             def phi_A_body_cm_np(P_b_cm):
#                 return self.A_np_fn(P_b_cm, params=A)
#         else:
#             def phi_A_body_cm_np(P_b_cm):
#                 return self.A_np_fn(P_b_cm)

#         def phi_A_world_m_np(P_w_m):
#             Pw    = np.asarray(P_w_m, float).reshape(-1, 3)
#             Pb_m  = (Pw - p_w) @ R_wb.T
#             Pb_cm = 100.0 * Pb_m
#             return 0.01 * phi_A_body_cm_np(Pb_cm)

#         phi_B_world_m_np = self.B_np_fn  # may be None

#         # --- Autosize root bbox around A in WORLD m (once) ---
#         def autosize_root_bbox():
#             dirs = []
#             s = 1/np.sqrt(2); t = 1/np.sqrt(3)
#             axes  = [[+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]]
#             edges = [[+s,+s,0],[+s,-s,0],[-s,+s,0],[-s,-s,0],[+s,0,+s],[+s,0,-s],[-s,0,+s],[-s,0,-s],
#                      [0,+s,+s],[0,+s,-s],[0,-s,+s],[0,-s,-s]]
#             corners = [[+t,+t,+t],[+t,+t,-t],[+t,-t,+t],[+t,-t,-t],[-t,+t,+t],[-t,+t,-t],[-t,-t,+t],[-t,-t,-t]]
#             for d in axes + edges + corners:
#                 v = np.asarray(d, float); v /= np.linalg.norm(v); dirs.append(v)
#             radii = np.linspace(0.05, 0.80, 32)  # m
#             max_hit, margin = 0.25, 0.05
#             for d in dirs:
#                 P  = p_w[None, :] + radii[:, None] * d[None, :]
#                 ph = phi_A_world_m_np(P)
#                 idx = np.argmax(ph > 0.01) if (ph > 0.01).any() else (len(radii) - 1)
#                 hit_r = radii[min(idx + 1, len(radii) - 1)]
#                 max_hit = max(max_hit, float(hit_r))
#             half = max_hit + margin
#             return (p_w[0] - half, p_w[0] + half,
#                     p_w[1] - half, p_w[1] + half,
#                     p_w[2] - half, p_w[2] + half)

#         if self._root_bounds is None:
#             self._root_bounds = autosize_root_bbox()

#         # --- Voxel structure (NumPy-only) ---
#         class Vox:
#             def __init__(self, nodes, L, b):
#                 self.nodes = np.asarray(nodes, np.int64).reshape(-1, 3)
#                 self.L = int(L)
#                 self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = map(float, b)
#             @classmethod
#             def root(cls, b):
#                 return cls(np.array([[0, 0, 0]], np.int64), 0, b)
#             def _sizes(self):
#                 W = self.xmax - self.xmin; H = self.ymax - self.ymin; D = self.zmax - self.zmin
#                 n = 1 << self.L
#                 dx, dy, dz = W / n, H / n, D / n
#                 return dx, dy, dz, 0.5*dx, 0.5*dy, 0.5*dz
#             def centers(self):
#                 dx, dy, dz, _, _, _ = self._sizes()
#                 ijk = self.nodes.astype(float)
#                 cx = self.xmin + (ijk[:, 0] + 0.5) * dx
#                 cy = self.ymin + (ijk[:, 1] + 0.5) * dy
#                 cz = self.zmin + (ijk[:, 2] + 0.5) * dz
#                 return np.column_stack([cx, cy, cz]).astype(np.float32, copy=False)
#             def half_sizes(self):
#                 _, _, _, hx, hy, hz = self._sizes()
#                 return float(hx), float(hy), float(hz)
#             def radius(self):
#                 hx, hy, hz = self.half_sizes()
#                 return float(np.linalg.norm([hx, hy, hz]))
#             def subdivide(self):
#                 i, j, k = self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2]
#                 kids = np.array([np.stack(((i<<1)|a, (j<<1)|b, (k<<1)|c), 1)
#                                  for a in (0, 1) for b in (0, 1) for c in (0, 1)]).reshape(-1, 3)
#                 return Vox(kids, self.L + 1,
#                            (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax))
#             def restrict(self, mask):
#                 self.nodes = self.nodes[np.asarray(mask, bool)]

#         # --- Refinement ---
#         vox = Vox.root(self._root_bounds)
#         for _ in range(self.max_refine):
#             vox = vox.subdivide()
#             r   = vox.radius()
#             C   = vox.centers()
#             keep1 = (phi_A_world_m_np(C) <= r)  # Phase-1 on A
#             vox.restrict(keep1)
#             if self.enable_phase2 and len(vox.nodes) > 0 and (phi_B_world_m_np is not None):
#                 C2   = vox.centers()
#                 phiB = phi_B_world_m_np(C2)
#                 min_u = np.min(phiB) + r
#                 keep2 = (phiB <= (min_u + r))
#                 vox.restrict(keep2)

#         C_final = vox.centers()
#         hx, hy, hz = vox.half_sizes()
#         r_cell = float(np.linalg.norm([hx, hy, hz]))  # m

#         # --- Deterministic pool from current frame (no scores) ---
#         if len(C_final) == 0:
#             C_pool_new = np.zeros((self.K_pool, 3), float)
#         else:
#             # Sort by coordinates for determinism
#             order_xyz = np.lexsort((C_final[:,2], C_final[:,1], C_final[:,0]))
#             C_sorted  = C_final[order_xyz]

#             # Evenly spaced subsample to K_pool for coverage (deterministic)
#             if len(C_sorted) > self.K_pool:
#                 idx = np.linspace(0, len(C_sorted) - 1, self.K_pool).astype(int)
#                 C_pool_new = C_sorted[idx]
#             else:
#                 need = self.K_pool - len(C_sorted)
#                 pad  = C_sorted[-1:] if len(C_sorted) > 0 else np.zeros((1,3), float)
#                 C_pool_new = np.vstack([C_sorted, np.repeat(pad, need, axis=0)]) if need > 0 else C_sorted

#         # --- Union with previous set (mapped to WORLD), dedup by voxel index ---
#         if self._cache is not None:
#             C_old_b_cm  = self._cache['C_b_cm']                  # (K,3) BODY cm from last iter
#             C_old_world = (C_old_b_cm / 100.0) @ R_wb.T + p_w     # map to WORLD at current pose
#             C_union = np.vstack([C_pool_new, C_old_world])

#             # Deduplicate using voxel indices at this final level (stable)
#             dx, dy, dz, _, _, _ = vox._sizes()
#             n = 1 << vox.L
#             ii = np.clip(np.floor((C_union[:,0] - vox.xmin)/dx).astype(np.int64), 0, n-1)
#             jj = np.clip(np.floor((C_union[:,1] - vox.ymin)/dy).astype(np.int64), 0, n-1)
#             kk = np.clip(np.floor((C_union[:,2] - vox.zmin)/dz).astype(np.int64), 0, n-1)
#             keys = ii*(n*n) + jj*n + kk
#             _, first_idx = np.unique(keys, return_index=True)  # keep first occurrence
#             keep_idx = np.sort(first_idx)
#             C_union  = C_union[keep_idx]
#         else:
#             C_union = C_pool_new

#         # --- Final deterministic take of K points (even subsample of sorted coords) ---
#         order_xyz = np.lexsort((C_union[:,2], C_union[:,1], C_union[:,0]))
#         C_sorted  = C_union[order_xyz]
#         if len(C_sorted) >= self.K:
#             idx = np.linspace(0, len(C_sorted) - 1, self.K).astype(int)
#             C_top_world = C_sorted[idx]
#         else:
#             need = self.K - len(C_sorted)
#             pad  = C_sorted[-1:] if len(C_sorted) > 0 else np.zeros((1,3), float)
#             C_top_world = np.vstack([C_sorted, np.repeat(pad, need, axis=0)]) if need > 0 else C_sorted

#         # Convert to BODY frame (cm), output, and cache
#         C_b_m  = (C_top_world - p_w[None, :]) @ R_bw.T
#         C_b_cm = (100.0 * C_b_m).astype(float)
#         r_out  = np.array([r_cell], float)

#         out['C_b_cm'] = C_b_cm
#         out['r_cell'] = r_out
#         self._cache   = {'C_b_cm': C_b_cm.copy(), 'r_cell': r_out.copy()}

        
#     def compute_derivatives(self, *_):
#         # Selection is non-smooth; no derivatives through the op.
#         pass



# def collision_check(
#     phi_A_body_cm_np,           # drone_sdf_np (NumPy; BODY cm)
#     phi_B_world_m,              # CSDL evaluator (WORLD m)
#     pos_w_m, quat_wxyz,         # CSDL vars
#     *, K=5, max_refine=8, enable_phase2=False, phi_B_world_m_np=None
# ):
#     # Feed pose into the op; (no A params for now)
#     inputs = csdl.VariableGroup()
#     inputs.pos_w_m, inputs.quat_wxyz = pos_w_m, quat_wxyz

#     op = BroadphaseOp(
#     A_np_fn=phi_A_body_cm_np,
#     B_np_fn=phi_B_world_m_np,
#     K=K, max_refine=max_refine, enable_phase2=enable_phase2
#     )
#     outs = op.evaluate(inputs)
#     C_b_cm, r_cell = outs.C_b_cm, outs.r_cell[0]   # (K,3), scalar

#     # Narrow phase 
#     def quat_to_R(q):
#         qw,qx,qy,qz = q[0],q[1],q[2],q[3]
#         r00 = 1-2*(qy*qy+qz*qz); r01 = 2*(qx*qy-qw*qz); r02 = 2*(qx*qz+qw*qy)
#         r10 = 2*(qx*qy+qw*qz);   r11 = 1-2*(qx*qx+qz*qz); r12 = 2*(qy*qz-qw*qx)
#         r20 = 2*(qx*qz-qw*qy);   r21 = 2*(qy*qz+qw*qx);   r22 = 1-2*(qx*qx+qy*qy)
#         return csdl.reshape(csdl.vstack((r00,r01,r02,r10,r11,r12,r20,r21,r22)), (3,3))

#     C_b_m  = C_b_cm / 100.0
#     R_wb   = quat_to_R(quat_wxyz)
#     C_rot  = csdl.matmat(C_b_m, csdl.transpose(R_wb))
#     Kk     = C_b_m.shape[0]
#     pos_t  = csdl.expand(pos_w_m, out_shape=(Kk,3), action='i->ji')
#     C_w    = pos_t + C_rot                           # (K,3) WORLD m

#     phiB_K = phi_B_world_m(C_w)                      # (K,)
#     d_soft = csdl.minimum(phiB_K - r_cell)           # conservative soft-min
#     return d_soft


def broadphase(
    phi_A_body_cm_np: Callable[[np.array], np.array],
    phi_B_world_m_np: Callable[[np.array], np.array],
    pos_w_m: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    max_refine: int = 8,
    enable_phase2: bool = True,
    K: int = 5,
    store_history: bool = False,
    t_idx: int | None = None,
    stash: dict | None = None,
    ):
     
   
    def _quat_to_R_wb(q):
        """Rotation matrix (WORLD ← BODY) from quaternion (w,x,y,z)."""
        w, x, y, z = q
        n = np.linalg.norm(q)
        if n == 0:
            return np.eye(3)
        w, x, y, z = q / n
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ])

    p_w  = pos_w_m
    R_wb = _quat_to_R_wb(quat_wxyz)

    def phi_A_numeric_world(P_w_m: np.ndarray) -> np.ndarray:
        P_w_m = np.asarray(P_w_m, float).reshape(-1, 3)
        P_b_m  = (P_w_m - p_w) @ R_wb
        P_b_cm = 100.0 * P_b_m
        return 0.01 * phi_A_body_cm_np(P_b_cm)
        #vals_cm = phi_A_body_cm_np(P_b_cm)
        #return 0.01 * np.asarray(vals_cm).reshape(-1)  # meters
        


    def phi_B_numeric_world(P_w_m: np.ndarray) -> np.ndarray:
        P_w_m = np.asarray(P_w_m, float).reshape(-1, 3)
        return phi_B_world_m_np(P_w_m)

    # -----------------------
    # Voxel structure
    # -----------------------
    class Voxels:
        def __init__(self, nodes: np.ndarray, L: int, bounds: tuple):
            self.nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
            self.L = int(L)
            self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = map(float, bounds)

        @classmethod
        def root(cls, bounds):
            return cls(nodes=np.array([[0,0,0]], dtype=np.int64), L=0, bounds=bounds)

        def _sizes(self):
            W = self.xmax - self.xmin; H = self.ymax - self.ymin; D = self.zmax - self.zmin
            n = 1 << self.L
            dx, dy, dz = W/n, H/n, D/n
            hx, hy, hz = 0.5*dx, 0.5*dy, 0.5*dz
            return dx, dy, dz, hx, hy, hz

        def centers(self) -> np.ndarray:
            dx, dy, dz, _, _, _ = self._sizes()
            i = self.nodes[:,0].astype(float); j = self.nodes[:,1].astype(float); k = self.nodes[:,2].astype(float)
            cx = self.xmin + (i + 0.5) * dx
            cy = self.ymin + (j + 0.5) * dy
            cz = self.zmin + (k + 0.5) * dz
            return np.column_stack([cx, cy, cz]).astype(np.float32, copy=False)  # float32 saves RAM

        def half_sizes(self) -> tuple:
            _, _, _, hx, hy, hz = self._sizes()
            return float(hx), float(hy), float(hz)

        def radius(self) -> float:
            hx, hy, hz = self.half_sizes()
            return float(np.sqrt(hx*hx + hy*hy + hz*hz))

        def subdivide(self):
            i = self.nodes[:,0]; j = self.nodes[:,1]; k = self.nodes[:,2]
            kids = np.array([
                np.stack(((i<<1)|0, (j<<1)|0, (k<<1)|0), axis=1),
                np.stack(((i<<1)|1, (j<<1)|0, (k<<1)|0), axis=1),
                np.stack(((i<<1)|0, (j<<1)|1, (k<<1)|0), axis=1),
                np.stack(((i<<1)|1, (j<<1)|1, (k<<1)|0), axis=1),
                np.stack(((i<<1)|0, (j<<1)|0, (k<<1)|1), axis=1),
                np.stack(((i<<1)|1, (j<<1)|0, (k<<1)|1), axis=1),
                np.stack(((i<<1)|0, (j<<1)|1, (k<<1)|1), axis=1),
                np.stack(((i<<1)|1, (j<<1)|1, (k<<1)|1), axis=1),
            ]).reshape(-1, 3)
            return Voxels(kids, self.L+1, (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax))

        def restrict(self, mask: np.ndarray):
            self.nodes = self.nodes[np.asarray(mask, bool)]

    # -----------------------
    # Auto-size root bounds by probing φ_A in world space
    # -----------------------
    def _autosize_root_bbox() -> tuple:
        dirs = []
        s = 1/np.sqrt(2); t = 1/np.sqrt(3)
        axes  = [[+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]]
        edges = [[+s,+s,0],[+s,-s,0],[-s,+s,0],[-s,-s,0],
                 [+s,0,+s],[+s,0,-s],[-s,0,+s],[-s,0,-s],
                 [0,+s,+s],[0,+s,-s],[0,-s,+s],[0,-s,-s]]
        corners = [[+t,+t,+t],[+t,+t,-t],[+t,-t,+t],[+t,-t,-t],
                   [-t,+t,+t],[-t,+t,-t],[-t,-t,+t],[-t,-t,-t]]
        for d in axes + edges + corners:
            v = np.asarray(d, float); v /= np.linalg.norm(v); dirs.append(v)
        radii = np.linspace(0.05, 0.80, 32)  # meters
        max_hit = 0.25; margin = 0.05
        for d in dirs:
            P = p_w[None, :] + radii[:, None] * np.asarray(d, float)[None, :]
            phis = phi_A_numeric_world(P)
            if (phis > 0.01).any():
                idx = int(np.argmax(phis > 0.01))
                hit_r = radii[min(idx+1, len(radii)-1)]
            else:
                hit_r = radii[-1]
            max_hit = max(max_hit, float(hit_r))
        half = max_hit + margin
        return (p_w[0]-half, p_w[0]+half, p_w[1]-half, p_w[1]+half, p_w[2]-half, p_w[2]+half)

    root_bounds = _autosize_root_bbox()

    # -----------------------
    # One refinement step
    # -----------------------
    def _refine_once(vox: "Voxels"):
        vox = vox.subdivide()
        r   = vox.radius()
        C   = vox.centers()
        keep1 = (phi_A_numeric_world(C) <= r)     # Phase-1 on φ_A
        vox.restrict(keep1)

        interval = (np.nan, np.nan)
        if enable_phase2 and len(vox.nodes) > 0:
            C2   = vox.centers()
            phiB = phi_B_numeric_world(C2)
            min_u = np.min(phiB) + r
            keep2 = (phiB <= (min_u + r))         # Phase-2 on φ_B
            vox.restrict(keep2)
            interval = (min_u - 2.0*r, min_u)
        return vox, interval

    # -----------------------
    # Run refinement (keep only final level unless store_history/plot)
    # -----------------------
    vox = Voxels.root(root_bounds)
    if store_history:
        kept3d     = {0: vox.centers().copy()}
        halfsizes  = {0: vox.half_sizes()}
        intervals  = {0: (np.nan, np.nan)}
    else:
        kept3d, halfsizes, intervals = {}, {}, None

    for L in range(max_refine):
        vox, inter = _refine_once(vox)

        if store_history:
            kept3d[L+1]    = vox.centers().copy()
            halfsizes[L+1] = vox.half_sizes()
            if intervals is not None:
                intervals[L+1] = inter
        else:
            if L + 1 == max_refine:
                kept3d    = {max_refine: vox.centers().copy()}
                halfsizes = {max_refine: vox.half_sizes()}


    # -----------------------
    # Select top-K candidates
    # -----------------------
    C_final = kept3d[max_refine]
    hx, hy, hz = halfsizes[max_refine]
    r_cell = float(np.linalg.norm([hx, hy, hz]))  # voxel half-diagonal

    phiB_centers = phi_B_numeric_world(C_final)
    scores = phiB_centers - r_cell
    order  = np.argsort(scores)
    k = int(min(K, len(order)))
    C_top = C_final[order[:k]]

        # ---- stash world-frame boxes for later viz (if requested) ----
    if stash is not None and t_idx is not None:
        # allow t_idx to be an int or a (gate_idx, time_idx) tuple
        key = t_idx if isinstance(t_idx, (tuple, list)) else int(t_idx)
        stash[key] = {
            "centers_world": C_top,                                  # (k,3) float32
            "half":           np.array([hx, hy, hz], np.float32),    # (3,)
            "r_cell":         float(r_cell),
            "score":          scores[order[:k]].astype(np.float32, copy=False),
            "K_req":          int(K),
            "K_sel":          k,
        }


    # Precompute candidate positions in BODY frame (constants in online stage)

    R_wb_np = _quat_to_R_wb(quat_wxyz)
    C_b_m  = (C_top - pos_w_m) @ R_wb_np
    C_b_cm = 100.0 * C_b_m

    return C_b_cm

# ================ CSDL Collision Metric ==============#
def collision_check(
    phi_A_body_cm: Callable[[csdl.Variable], csdl.Variable],
    phi_B_world_m: Callable[[csdl.Variable], csdl.Variable],
    pos_w_m: csdl.Variable,
    quat_wxyz: csdl.Variable,
    C_b_cm: csdl.Variable,
) -> csdl.Variable:
   

    # -----------------------
    # CSDL helpers (narrow-phase)
    # -----------------------
    def quat_to_rm_csdl(q: csdl.Variable) -> csdl.Variable:
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        r00 = 1.0 - 2.0*(qy*qy + qz*qz)
        r01 = 2.0*(qx*qy - qw*qz)
        r02 = 2.0*(qx*qz + qw*qy)
        r10 = 2.0*(qx*qy + qw*qz)
        r11 = 1.0 - 2.0*(qx*qx + qz*qz)
        r12 = 2.0*(qy*qz - qw*qx)
        r20 = 2.0*(qx*qz - qw*qy)
        r21 = 2.0*(qy*qz + qw*qx)
        r22 = 1.0 - 2.0*(qx*qx + qy*qy)
        R_flat = csdl.vstack((r00, r01, r02, r10, r11, r12, r20, r21, r22))
        return csdl.reshape(R_flat, (3, 3))

    # ---------- BODY(cm) -> WORLD(m) wrapper ----------
    def world_to_body_cm(P_w_m: csdl.Variable,
                        pos_w_m: csdl.Variable,
                        quat_wxyz: csdl.Variable) -> csdl.Variable:
        R_bw = csdl.transpose(quat_to_rm_csdl(quat_wxyz))  # body_from_world
        if P_w_m.shape == (3,):
            return 100.0 * csdl.matvec(R_bw, (P_w_m - pos_w_m))
        Np = P_w_m.shape[0]
        out = csdl.Variable(value=np.zeros((Np, 3)))
        for i in csdl.frange(Np):
            vi = csdl.matvec(R_bw, (P_w_m[i, :] - pos_w_m))
            out = out.set(csdl.slice[i, :], 100.0 * vi)
        return out

    def make_phi_A_world_m(phi_A_body_cm_fn,
                        pos_w_m: csdl.Variable,
                        quat_wxyz: csdl.Variable):
        def phi_A_world_m(P_w_m: csdl.Variable) -> csdl.Variable:
            P_b_cm = world_to_body_cm(P_w_m, pos_w_m, quat_wxyz)
            return 0.01 * phi_A_body_cm_fn(P_b_cm)  # cm -> m
        return phi_A_world_m

    # Pose-tied candidate positions (WORLD) from body-frame constants
    C_b_cm_cs = C_b_cm        # (K,3) constants in cm
    C_b_m_cs  = C_b_cm_cs / 100.0                  # (K,3) meters
    R_wb_cs   = quat_to_rm_csdl(quat_wxyz)         # WORLD←BODY rotation
    C_rot     = csdl.matmat(C_b_m_cs, csdl.transpose(R_wb_cs))  # (K,3)
    KK        = C_b_m_cs.shape[0]
    pos_tiled = csdl.expand(pos_w_m, out_shape=(KK,3), action='i->ji')  # (K,3)
    C_w_cs    = pos_tiled + C_rot                  # (K,3) WORLD candidates

    # === Evaluate both SDFs in WORLD(m) at the K candidates ===
    phiB_K = phi_B_world_m(C_w_cs)                 # (K,) or (K,1) — unchanged
    phi_A_world_m = make_phi_A_world_m(phi_A_body_cm, pos_w_m, quat_wxyz)
    phiA_K = phi_A_world_m(C_w_cs)                 # (K,) or (K,1) — unchanged

    # Conservative aggregator (example)
    #m_K   = phiB_K - r_cell
    m_K  = phiB_K + phiA_K
    d_soft = csdl.minimum(m_K)


    return d_soft










# _DEF_EPS = 1e-12

# ================ HYBRID METHOD ================
# def _as_array(x, shape=None):
#     arr = np.asarray(getattr(x, 'value', x), float)
#     return arr if shape is None else arr.reshape(shape)

# def _quat_to_R_wb(q):
#     """Rotation (WORLD ← BODY) from quaternion q=[w,x,y,z] (numpy)."""
#     w, x, y, z = q
#     n = np.linalg.norm(q)
#     if n == 0:
#         return np.eye(3)
#     w, x, y, z = q / n
#     return np.array([
#         [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
#         [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
#         [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
#     ])

# def quat_to_rm_csdl(q: csdl.Variable) -> csdl.Variable:
#     """Quaternion [w,x,y,z] -> 3x3 rotation (CSDL). Normalizes internally."""
#     qw, qx, qy, qz = q[0], q[1], q[2], q[3]
#     nrm = csdl.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + _DEF_EPS
#     qw, qx, qy, qz = qw/nrm, qx/nrm, qy/nrm, qz/nrm
#     r00 = 1.0 - 2.0*(qy*qy + qz*qz)
#     r01 = 2.0*(qx*qy - qw*qz)
#     r02 = 2.0*(qx*qz + qw*qy)
#     r10 = 2.0*(qx*qy + qw*qz)
#     r11 = 1.0 - 2.0*(qx*qx + qz*qz)
#     r12 = 2.0*(qy*qz - qw*qx)
#     r20 = 2.0*(qx*qz - qw*qy)
#     r21 = 2.0*(qy*qz + qw*qx)
#     r22 = 1.0 - 2.0*(qx*qx + qy*qy)
#     R_flat = csdl.vstack((r00, r01, r02, r10, r11, r12, r20, r21, r22))
#     return csdl.reshape(R_flat, (3, 3))

# # ---------- main entry ----------
# def collision_check(
#     phi_A_body_cm: Callable[[csdl.Variable], csdl.Variable],
#     phi_B_world_m: Callable[[csdl.Variable], csdl.Variable],
#     pos_w_m: csdl.Variable,
#     quat_wxyz: csdl.Variable,
#     *,
#     b_seed_center_w_m: np.ndarray,    # e.g., gate center
#     max_refine: int = 8,
#     enable_phase2: bool = True,
#     plot_octree: bool = True,
#     plot_kkt: bool = True,
#     gamma: float = 50.0,
#     newton_tol: float = 1e-8,
#     recorder: Optional[csdl.Recorder] = None
# ) -> Dict[str, object]:
#     """
#     Returns a dict with:
#       - keptA, halfA, intervalsA, keptB, halfB, intervalsB (by level)
#       - mnn_pairs (final), C_A_top (1,3), C_B_top (1,3)
#       - KKT: m, F_star, a, b, pair_gap  (csdl.Variables)
#     """

#     # =============== OFFLINE: dual-sided octree + MNN (pause main recorder) ===============
#     if recorder is not None:
#         recorder.stop()
#     dummy = csdl.Recorder(inline=True); dummy.start()

#     # pose (numpy)
#     p_w  = _as_array(pos_w_m, (3,))
#     R_wb = _quat_to_R_wb(_as_array(quat_wxyz, (4,)))

#     # numeric wrappers
#     def phi_A_numeric_world(P_w_m: np.ndarray) -> np.ndarray:
#         P_w_m = np.asarray(P_w_m, float).reshape(-1, 3)
#         P_b_m  = (P_w_m - p_w) @ R_wb.T
#         P_b_cm = 100.0 * P_b_m
#         vals_cm = phi_A_body_cm(csdl.Variable(value=P_b_cm)).value
#         return 0.01 * np.asarray(vals_cm).reshape(-1)

#     def phi_B_numeric_world(P_w_m: np.ndarray) -> np.ndarray:
#         P_w_m = np.asarray(P_w_m, float).reshape(-1, 3)
#         vals = phi_B_world_m(csdl.Variable(value=P_w_m)).value
#         return np.asarray(vals).reshape(-1)

#     # voxel structure
#     class Voxels:
#         def __init__(self, nodes: np.ndarray, L: int, bounds):
#             self.nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
#             self.L = int(L)
#             self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = map(float, bounds)
#         @classmethod
#         def root(cls, bounds):
#             return cls(nodes=np.array([[0,0,0]]), L=0, bounds=bounds)
#         def _sizes(self):
#             W = self.xmax - self.xmin; H = self.ymax - self.ymin; D = self.zmax - self.zmin
#             n = 1 << self.L
#             dx, dy, dz = W/n, H/n, D/n
#             hx, hy, hz = 0.5*dx, 0.5*dy, 0.5*dz
#             return dx, dy, dz, hx, hy, hz
#         def centers(self):
#             dx, dy, dz, _, _, _ = self._sizes()
#             i = self.nodes[:,0].astype(float); j = self.nodes[:,1].astype(float); k = self.nodes[:,2].astype(float)
#             cx = self.xmin + (i + 0.5) * dx
#             cy = self.ymin + (j + 0.5) * dy
#             cz = self.zmin + (k + 0.5) * dz
#             return np.column_stack([cx, cy, cz])
#         def half_sizes(self):
#             _, _, _, hx, hy, hz = self._sizes()
#             return float(hx), float(hy), float(hz)
#         def radius(self):
#             hx, hy, hz = self.half_sizes()
#             return float(np.sqrt(hx*hx + hy*hy + hz*hz))
#         def subdivide(self):
#             i = self.nodes[:,0]; j = self.nodes[:,1]; k = self.nodes[:,2]
#             kids = np.array([
#                 np.stack(((i<<1)|0, (j<<1)|0, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|0, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|0, (j<<1)|1, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|1, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|0, (j<<1)|0, (k<<1)|1), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|0, (k<<1)|1), axis=1),
#                 np.stack(((i<<1)|0, (j<<1)|1, (k<<1)|1), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|1, (k<<1)|1), axis=1),
#             ]).reshape(-1, 3)
#             return Voxels(kids, self.L+1, (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax))
#         def restrict(self, mask):
#             self.nodes = self.nodes[np.asarray(mask, bool)]

#     # robust autosize (minimal tweak): ensure half ≥ |φ(seed)| + margin
#     def _autosize_root_bbox_from(phi_fn_numeric, seed_center, hit_thresh=0.01, ray_max=0.80, margin=0.05):
#         dirs = []
#         s = 1/np.sqrt(2); t = 1/np.sqrt(3)
#         axes  = [[+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]]
#         edges = [[+s,+s,0],[+s,-s,0],[-s,+s,0],[-s,-s,0],
#                  [+s,0,+s],[+s,0,-s],[-s,0,+s],[-s,0,-s],
#                  [0,+s,+s],[0,+s,-s],[0,-s,+s],[0,-s,-s]]
#         corners = [[+t,+t,+t],[+t,+t,-t],[+t,-t,+t],[+t,-t,-t],
#                    [-t,+t,+t],[-t,+t,-t],[-t,-t,+t],[-t,-t,-t]]
#         for d in axes + edges + corners:
#             v = np.asarray(d, float); v /= np.linalg.norm(v); dirs.append(v)

#         radii = np.linspace(0.05, ray_max, 32)
#         max_hit = 0.25
#         for d in dirs:
#             points = seed_center[None,:] + radii[:,None]*d[None,:]
#             phis = phi_fn_numeric(points)
#             if (phis > hit_thresh).any():
#                 idx = np.argmax(phis > hit_thresh)
#                 hit_r = radii[min(idx+1, len(radii)-1)]
#             else:
#                 hit_r = radii[-1]
#             max_hit = max(max_hit, float(hit_r))

#         half = max_hit + margin
#         phi0 = float(phi_fn_numeric(np.asarray(seed_center, float).reshape(1, -1))[0])
#         half = max(half, abs(phi0) + margin)

#         return (seed_center[0]-half, seed_center[0]+half,
#                 seed_center[1]-half, seed_center[1]+half,
#                 seed_center[2]-half, seed_center[2]+half)

#     # A-centric (seed at drone) and B-centric (seed at gate center)
#     rootA = _autosize_root_bbox_from(phi_A_numeric_world, p_w)
#     b_seed = np.asarray(b_seed_center_w_m, float).reshape(3,)
#     rootB = _autosize_root_bbox_from(phi_B_numeric_world, b_seed)

#     # one refinement step (A side)
#     def _refine_once_A(vox: "Voxels"):
#         vox = vox.subdivide()
#         r   = vox.radius()
#         C   = vox.centers()
#         keep1 = (phi_A_numeric_world(C) <= r)                    # Phase-1(A)
#         vox.restrict(keep1)
#         interval = (np.nan, np.nan)
#         if enable_phase2 and len(vox.nodes) > 0:
#             C2   = vox.centers()
#             phiB = phi_B_numeric_world(C2)
#             min_u = np.min(phiB) + r                              # “old” +r then +r → +2r window
#             keep2 = (phiB <= (min_u + r))                         # Phase-2(A)
#             vox.restrict(keep2)
#             interval = (min_u - 2.0*r, min_u)
#         return vox, interval

#     # one refinement step (B side)
#     def _refine_once_B(vox: "Voxels"):
#         vox = vox.subdivide()
#         r   = vox.radius()
#         C   = vox.centers()
#         keep1 = (phi_B_numeric_world(C) <= r)                    # Phase-1(B)
#         vox.restrict(keep1)
#         interval = (np.nan, np.nan)
#         if enable_phase2 and len(vox.nodes) > 0:
#             C2   = vox.centers()
#             phiA = phi_A_numeric_world(C2)
#             min_u = np.min(phiA) + r
#             keep2 = (phiA <= (min_u + r))                         # Phase-2(B)
#             vox.restrict(keep2)
#             interval = (min_u - 2.0*r, min_u)
#         return vox, interval

#     # run refinements
#     voxA = Voxels.root(rootA)
#     keptA, intervalsA, halfA = {0: voxA.centers().copy()}, {0: (np.nan, np.nan)}, {0: voxA.half_sizes()}
#     for L in range(max_refine):
#         voxA, interA = _refine_once_A(voxA)
#         keptA[L+1]      = voxA.centers().copy()
#         intervalsA[L+1] = interA
#         halfA[L+1]      = voxA.half_sizes()

#     voxB = Voxels.root(rootB)
#     keptB, intervalsB, halfB = {0: voxB.centers().copy()}, {0: (np.nan, np.nan)}, {0: voxB.half_sizes()}
#     for L in range(max_refine):
#         voxB, interB = _refine_once_B(voxB)
#         keptB[L+1]      = voxB.centers().copy()
#         intervalsB[L+1] = interB
#         halfB[L+1]      = voxB.half_sizes()

#     # final clouds
#     C_A_final = keptA[max_refine]
#     C_B_final = keptB[max_refine]
#     hxA, hyA, hzA = halfA[max_refine]
#     hxB, hyB, hzB = halfB[max_refine]
#     rA = float(np.linalg.norm([hxA, hyA, hzA]))
#     rB = float(np.linalg.norm([hxB, hyB, hzB]))

#     # MNN pairs
#     mnn_pairs: List[Tuple[int,int]] = []
#     if len(C_A_final) > 0 and len(C_B_final) > 0:
#         d2 = np.sum((C_A_final[:,None,:] - C_B_final[None,:,:])**2, axis=2)
#         nnB_of_A = np.argmin(d2, axis=1)
#         nnA_of_B = np.argmin(d2, axis=0)
#         for iA, jB in enumerate(nnB_of_A):
#             if nnA_of_B[jB] == iA:
#                 mnn_pairs.append((iA, jB))

#     # rank and take top-1
#     def _pair_score(iA, jB):
#         a = phi_A_numeric_world(C_A_final[iA:iA+1])[0]
#         b = phi_B_numeric_world(C_B_final[jB:jB+1])[0]
#         dist = float(np.linalg.norm(C_A_final[iA] - C_B_final[jB]))
#         return np.sqrt(a*a + b*b) + dist - (rA + rB)

#     C_A_top = np.zeros((0,3)); C_B_top = np.zeros((0,3)); top_pair = None
#     if mnn_pairs:
#         scores = np.array([_pair_score(iA, jB) for (iA, jB) in mnn_pairs], float)
#         order  = np.argsort(scores)
#         idx0   = int(order[0])
#         top_pair = mnn_pairs[idx0]
#         C_A_top = C_A_final[[top_pair[0]]]
#         C_B_top = C_B_final[[top_pair[1]]]

#     # restore main recorder for online KKT
#     dummy.stop()
#     if recorder is not None:
#         recorder.start()

#     # optional OCTREE plotting
#     if plot_octree:
#         try:
#             import pyvista as pv
#             # bounds union helper
#             def _cloud_bounds(C, h):
#                 if len(C)==0: return None
#                 hx, hy, hz = h
#                 H = np.tile(np.array([hx, hy, hz], float), (len(C), 1))
#                 mins = C - H; maxs = C + H
#                 return mins.min(axis=0), maxs.max(axis=0)

#             bA = _cloud_bounds(keptA[max_refine], halfA[max_refine])
#             bB = _cloud_bounds(keptB[max_refine], halfB[max_refine])
#             if bA and bB:
#                 mins = np.minimum(bA[0], bB[0]); maxs = np.maximum(bA[1], bB[1])
#             elif bA: mins, maxs = bA
#             elif bB: mins, maxs = bB
#             else:
#                 mins = np.array([rootA[0], rootA[2], rootA[4]])
#                 maxs = np.array([rootA[1], rootA[3], rootA[5]])
#             pad = 0.35
#             xmin, ymin, zmin = mins - pad; xmax, ymax, zmax = maxs + pad

#             Nx = Ny = Nz = 96
#             x = np.linspace(xmin, xmax, Nx)
#             y = np.linspace(ymin, ymax, Ny)
#             z = np.linspace(zmin, zmax, Nz)
#             X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
#             P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
#             phiA = phi_A_numeric_world(P).reshape(X.shape)
#             phiB = phi_B_numeric_world(P).reshape(X.shape)

#             plotter = pv.Plotter()
#             # φA=0
#             if (phiA.min() <= 0.0) and (phiA.max() >= 0.0):
#                 gridA = pv.ImageData()
#                 gridA.dimensions = np.array(phiA.shape)
#                 gridA.spacing   = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
#                 gridA.origin    = (x[0], y[0], z[0])
#                 gridA.point_data["phiA"] = phiA.flatten(order="F")
#                 contA = gridA.contour(isosurfaces=[0.0], scalars="phiA")
#                 plotter.add_mesh(contA, color="royalblue", opacity=0.55)
#             # φB=0
#             if (phiB.min() <= 0.0) and (phiB.max() >= 0.0):
#                 gridB = pv.ImageData()
#                 gridB.dimensions = np.array(phiB.shape)
#                 gridB.spacing   = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
#                 gridB.origin    = (x[0], y[0], z[0])
#                 gridB.point_data["phiB"] = phiB.flatten(order="F")
#                 contB = gridB.contour(isosurfaces=[0.0], scalars="phiB")
#                 plotter.add_mesh(contB, color="tomato", opacity=0.55)

#             # wireframe voxels (no pair lines)
#             MAX_DRAW = 1500
#             CdrawA = keptA[max_refine]; stepA = max(1, len(CdrawA)//MAX_DRAW if len(CdrawA) else 1)
#             hxA, hyA, hzA = halfA[max_refine]
#             for i in range(0, len(CdrawA), stepA):
#                 cx, cy, cz = CdrawA[i]
#                 cube = pv.Cube(center=(cx, cy, cz),
#                                x_length=2*hxA, y_length=2*hyA, z_length=2*hzA)
#                 plotter.add_mesh(cube, style="wireframe", color="yellow", line_width=1.0, opacity=0.85)

#             CdrawB = keptB[max_refine]; stepB = max(1, len(CdrawB)//MAX_DRAW if len(CdrawB) else 1)
#             hxB, hyB, hzB = halfB[max_refine]
#             for i in range(0, len(CdrawB), stepB):
#                 cx, cy, cz = CdrawB[i]
#                 cube = pv.Cube(center=(cx, cy, cz),
#                                x_length=2*hxB, y_length=2*hyB, z_length=2*hzB)
#                 plotter.add_mesh(cube, style="wireframe", color="cyan", line_width=1.0, opacity=0.85)

#             plotter.show_bounds(grid='front', location='outer', color='black',
#                                 xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
#             plotter.add_axes(); plotter.add_title("Two-Sided Broad-Phase: A(yellow) / B(cyan)")
#             plotter.camera_position = 'iso'
#             plotter.show()
#         except Exception as e:
#             print("[octree viz] skipped:", e)

#         # level diagnostics
#         try:
#             import matplotlib.pyplot as plt
#             levels = sorted(keptA.keys())
#             cntA = [len(keptA[L]) for L in levels]
#             cntB = [len(keptB[L]) for L in levels]
#             lowsA  = [intervalsA[L][0] for L in levels]
#             uppsA  = [intervalsA[L][1] for L in levels]
#             midsA  = [0.5*(l+u) for l,u in zip(lowsA, uppsA)]
#             lowsB  = [intervalsB[L][0] for L in levels]
#             uppsB  = [intervalsB[L][1] for L in levels]
#             midsB  = [0.5*(l+u) for l,u in zip(lowsB, uppsB)]
#             fig, axs = plt.subplots(1,3, figsize=(14,4))
#             axs[0].plot(levels, cntA, marker='o', label='A kept')
#             axs[0].plot(levels, cntB, marker='s', label='B kept')
#             axs[0].set_xlabel("L"); axs[0].set_ylabel("# voxels kept"); axs[0].legend(); axs[0].grid(alpha=0.3)
#             axs[1].plot(levels, lowsA, marker='v', label='A lower')
#             axs[1].plot(levels, uppsA, marker='^', label='A upper')
#             axs[1].plot(levels, midsA, marker='o', linestyle='--', label='A mid')
#             axs[1].set_xlabel("L"); axs[1].set_ylabel("Bracket on min φ_B"); axs[1].legend(); axs[1].grid(alpha=0.3)
#             axs[2].plot(levels, lowsB, marker='v', label='B lower')
#             axs[2].plot(levels, uppsB, marker='^', label='B upper')
#             axs[2].plot(levels, midsB, marker='o', linestyle='--', label='B mid')
#             axs[2].set_xlabel("L"); axs[2].set_ylabel("Bracket on min φ_A"); axs[2].legend(); axs[2].grid(alpha=0.3)
#             plt.suptitle(f"MNN pairs @final: {len(mnn_pairs)} (top-1 shown: {len(C_A_top)})")
#             plt.show()
#         except Exception as e:
#             print("[octree plots] skipped:", e)

#     # =============== ONLINE: penalized KKT (seeded by top-1 pair) =========================
#     # CSDL SDFs in world (m)
#     def phi_A_world_m(p_w: csdl.Variable) -> csdl.Variable:
#         R_wb = quat_to_rm_csdl(quat_wxyz)
#         R_bw = csdl.transpose(R_wb)
#         if p_w.shape == (3,):
#             rel_w  = p_w - pos_w_m
#             x_b_cm = 100.0 * csdl.matvec(R_bw, rel_w)
#             return phi_A_body_cm(x_b_cm) / 100.0
#         K = p_w.shape[0]
#         pos_tiled = csdl.expand(pos_w_m, out_shape=(K,3), action='i->ji')
#         rel_w  = p_w - pos_tiled
#         x_b_cm = 100.0 * csdl.matmat(rel_w, csdl.transpose(R_bw))
#         return phi_A_body_cm(x_b_cm) / 100.0

#     # penalty KKT with independent seeds for a,b
#     def penalty_kkt_pair(phi_A, phi_B, a0_np, b0_np, gamma=50.0, newton_tol=1e-8, name="penalty_kkt"):
#         a = csdl.Variable(name=f"{name}:a", shape=(3,), value=a0_np)
#         b = csdl.Variable(name=f"{name}:b", shape=(3,), value=b0_np)
#         phiA = phi_A(a);  phiB = phi_B(b)
#         gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
#         gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))
#         R_a = (a - b) + gamma * phiA * gA
#         R_b = (b - a) + gamma * phiB * gB
#         solver = csdl.nonlinear_solvers.Newton(name, tolerance=newton_tol)
#         solver.add_state(a, R_a, initial_value=a0_np)
#         solver.add_state(b, R_b, initial_value=b0_np)
#         solver.run()
#         m = 0.5*(a+b)
#         F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.05
#         #diff = a - b
#         #gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)
#         return m, F_star, a, b

#     # choose seeds
#     if len(C_A_top) == 1:
#         a0 = C_A_top[0]; b0 = C_B_top[0]
#         # print("Phi_A_top:", phi_A_numeric_world(a0)); print("Phi_B_top:", phi_B_numeric_world(b0));
#         # print("Top-1 MNN pair:", a0, b0)
#         # print("Midpoint C:", 0.5*(a0+b0))
#         # --- pick gamma from seed balance ---
#         a0 = C_A_top[0]; b0 = C_B_top[0]
#         d0 = float(np.linalg.norm(a0 - b0))
#         phiA0 = phi_A_numeric_world(a0)
#         phiB0 = phi_B_numeric_world(b0)
#         phi_bar = max(1e-4, 0.5*(abs(phiA0) + abs(phiB0)))   # guard
#         gamma_bal = np.clip(d0 / phi_bar, 5.0, 300.0)        # seed-balanced gamma

#         m, F_star, a, b = penalty_kkt_pair(phi_A_world_m, phi_B_world_m,
#                                                     a0, b0,
#                                                     gamma=gamma_bal, newton_tol=newton_tol,
#                                                     name="collision_kkt_gbal")
#         # # --- (optional) two-stage continuation for robustness ---
#         # gammas = [0.5*gamma_bal, gamma_bal, 0.5*(gamma_bal+gamma), gamma]  # last is your target (e.g., 50)
#         # a_init, b_init = a0, b0
#         # for g in gammas:
#         #     m, F_star, a, b, pair_gap = penalty_kkt_pair(phi_A_world_m, phi_B_world_m,
#         #                                                 a_init, b_init,
#         #                                                 gamma=g, newton_tol=newton_tol,
#         #                                                 name=f"collision_kkt_g{g:.1f}")
#         #     # warm start next stage
#         #     a_init = np.asarray(a.value).reshape(3)
#         #     b_init = np.asarray(b.value).reshape(3)

#     else:
#         # fallback: midpoint between drone and gate center if no pairs
#         mid = 0.5*(p_w + b_seed)
#         print("Midpoint fallback:", mid)
#         m, F_star, a, b = penalty_kkt_pair(phi_A_world_m, phi_B_world_m, mid, mid,
#                                                      gamma=gamma, newton_tol=newton_tol, name="collision_kkt_fallback")

#     # optional KKT plotting
#     if plot_kkt:
#         try:
#             import pyvista as pv
#             # quick grids (drone tight; gate broader)
#             pos_np = np.asarray(pos_w_m.value)
#             d_half = 0.22
#             bounds_drone = ((pos_np[0]-d_half, pos_np[0]+d_half),
#                             (pos_np[1]-d_half, pos_np[1]+d_half),
#                             (pos_np[2]-d_half, pos_np[2]+d_half))
#             g_half = np.array([1.0, 1.5, 1.5])  # rough sphere/box max with margin
#             gcen  = b_seed
#             bounds_gate = ((gcen[0]-g_half[0], gcen[0]+g_half[0]),
#                            (gcen[1]-g_half[1], gcen[1]+g_half[1]),
#                            (gcen[2]-g_half[2], gcen[2]+g_half[2]))
#             def contour_zero(phi_fn, bounds, dx, scal_name):
#                 if not hasattr(pv, "ImageData"): raise RuntimeError("PyVista ImageData missing")
#                 (xmin,xmax),(ymin,ymax),(zmin,zmax) = bounds
#                 nx = int(np.floor((xmax-xmin)/dx))+1
#                 ny = int(np.floor((ymax-ymin)/dx))+1
#                 nz = int(np.floor((zmax-zmin)/dx))+1
#                 grid = pv.ImageData()
#                 grid.dimensions = (nx,ny,nz); grid.origin=(xmin,ymin,zmin); grid.spacing=(dx,dx,dx)
#                 P = np.asarray(grid.points)
#                 vals = phi_fn(csdl.Variable(value=P.astype(float))).value.reshape(-1)
#                 grid[scal_name] = vals
#                 return grid.contour(isosurfaces=[0.0], scalars=scal_name)

#             surf_A = contour_zero(phi_A_world_m, bounds_drone, 0.01, "phiA")
#             surf_B = contour_zero(phi_B_world_m, bounds_gate,  0.04, "phiB")

#             a_np = np.asarray(a.value).reshape(3)
#             b_np = np.asarray(b.value).reshape(3)
#             m_np = np.asarray(m.value).reshape(3)
#             s_a = pv.Sphere(radius=0.010, center=a_np)
#             s_b = pv.Sphere(radius=0.010, center=b_np)
#             s_m = pv.Sphere(radius=0.008, center=m_np)

#             p = pv.Plotter()
#             p.add_mesh(surf_A, color="#1f77b4", opacity=0.75, smooth_shading=True, label="Drone (A=0)")
#             p.add_mesh(surf_B, color="#f46c6c", opacity=0.35, smooth_shading=True, label="Gate (B=0)")
#             p.add_mesh(s_a, color="#bc2525"); p.add_mesh(s_b, color="#9467bd"); p.add_mesh(s_m, color="#ffdd57")
#             p.add_axes(line_width=2); p.add_legend(bcolor="black", border=True)
#             p.set_background("black", top="dimgray"); p.camera_position="iso"
#             p.show(title="Penalty KKT: φA=0, φB=0, witnesses a/b, midpoint m")
#         except Exception as e:
#             print("[kkt viz] skipped:", e)

#     # return dict(
#     #     keptA=keptA, halfA=halfA, intervalsA=intervalsA,
#     #     keptB=keptB, halfB=halfB, intervalsB=intervalsB,
#     #     mnn_pairs=mnn_pairs, C_A_top=C_A_top, C_B_top=C_B_top,
#     #     m=m, F_star=F_star, a=a, b=b, pair_gap=pair_gap
#     # )
#     return m, F_star, a, b


# =============== END HYBRID METHOD ===============




#SDF = Callable[[csdl.Variable], csdl.Variable]

#_DEF_EPS = 1e-12

# # Collision check similar to paper
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     rho: float = 20.0,             # higher = sharper max; start 5–20
#     return_all: bool = False,
#     newton_name: str = "collision_check_smooth",
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Differentiable SDF-SDF proximity/collision check

#     Steps:
#       1) Define smooth joint field F(x) = max(phi_A(x), phi_B(x)) 
#       2) Do K fixed 'pull' steps: x <- x - eta * ∇F / ||∇F||.
#       3) Project once from x_K to each surface:
#            a = x_K - phi_A(x_K) / ||∇phi_A(x_K)||^2 * ∇phi_A(x_K)
#            b = x_K - phi_B(x_K) / ||∇phi_B(x_K)||^2 * ∇phi_B(x_K)
#       4) Outputs: gap = ||a - b||  (always >= 0), F_star = F(x_K),
#          and optionally x_K, a, b.

#     Returns:
#       If return_all: (x_K, F_star, a, b, gap), else just F.
#     """
#     # ---- seed x0 as leaf variable ----
#     if isinstance(x0, csdl.Variable):
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     else:
#         x0_np = np.asarray(x0, float).reshape(3,)
#     #x = csdl.Variable(name=f"{newton_name}:x0", shape=(3,), value=x0_np)
#     x = x0
#     # ---- finite gradient steps on F(x) = max(phi_A, phi_B) ----

#     #for c_i in (0.9, 0.5, 0.25): 
#     #c_i = 1.0

#       # inside each pull iteration
#     FA = phi_A(x)
#     FB = phi_B(x)
#     F  = csdl.maximum(FA, FB, rho=rho)

#     gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))
#     gF_norm = csdl.norm(gF) + _DEF_EPS

#     # adaptive step length: eta = min(c * |F|, eta_max)
#     #eta = csdl.minimum(c_i * csdl.absolute(F), eta_max)   # choose c_i per pull, e.g. 0.7, 0.35, 0.2

#     x = x - (gF / gF_norm)

#     xK = x
#     F_star = csdl.maximum(phi_A(xK), phi_B(xK), rho=rho) -1.0

    
#     #---- two-step projection to each surface from xK ----
#     if return_all:
#         # ---------- helpers: safe L2 norm and unit ----------
#         def safe_norm2(v, eps=_DEF_EPS):
#             ax = csdl.absolute(v)
#             vmax01 = csdl.maximum(ax[0], ax[1])
#             vmax   = csdl.maximum(vmax01, ax[2])            # max(|v_i|)
#             scale  = csdl.maximum(vmax, csdl.Variable(value = 1.0))                # keep >= 1 to avoid div-by-zero
#             v_sc   = v / scale
#             return scale * csdl.sqrt(csdl.vdot(v_sc, v_sc) + eps)

#         def safe_unit(v, eps=_DEF_EPS):
#             return v / (safe_norm2(v, eps) + eps)

#         # ---- keep a copy of the candidate x* for residuals ----
#         x_cand = xK
#         np.set_printoptions(precision=6, suppress=True)

#         def _inf_norm(u):      # for numpy arrays (after .value)
#             return float(np.max(np.abs(u)))


#         x_cand = xK

#         # ===== A: first projection step =====
#         print("[PIN] before phi_A(x)")
#         FA0 = phi_A(x_cand)
#         print("[PIN] after  phi_A(x) | FA(x)=", float(FA0.value))

#         print("[PIN] before gradA(x)")
#         gA0 = csdl.reshape(csdl.derivative(ofs=FA0, wrts=x_cand), (3,))
#         print("[PIN] after  gradA(x) | max|gA|=", float(np.max(np.abs(gA0.value))))

#         gA0n = safe_norm2(gA0) + _DEF_EPS
#         sA   = csdl.absolute(FA0) / gA0n
#         a1   = x_cand - (FA0 * gA0) / (gA0n * gA0n)

#         print("[DBG] A0  | FA(x)       =", float(FA0.value),
#                 "||gA||    =", float(gA0n.value),
#                 "max|gA|   =", float(np.max(np.abs(gA0.value))),
#                 "step      =", float(sA.value),
#                 "||x||inf  =", _inf_norm(x_cand.value),
#                 "||a1||inf =", _inf_norm(a1.value))

#         # This call is a common overflow point if a1 jumped far:
#         FA1 = phi_A(a1)
#         print("[DBG] A1  | FA(a1)      =", float(FA1.value))

#         # ===== A: second projection step =====
#         gA1  = csdl.reshape(csdl.derivative(ofs=FA1, wrts=a1), (3,))
#         gA1n = safe_norm2(gA1) + _DEF_EPS
#         a    = a1 - (FA1 * gA1) / (gA1n * gA1n)

#         print("[DBG] A2  | ||gA(a1)||  =", float(gA1n.value),
#                 "max|gA(a1)|=", float(np.max(np.abs(gA1.value))),
#                 "||a||inf   =", _inf_norm(a.value))

#         # ===== B: first projection step =====
#         FB0  = phi_B(x_cand)
#         gB0  = csdl.reshape(csdl.derivative(ofs=FB0, wrts=x_cand), (3,))
#         gB0n = safe_norm2(gB0) + _DEF_EPS
#         sB   = csdl.absolute(FB0) / gB0n
#         b1   = x_cand - (FB0 * gB0) / (gB0n * gB0n)

#         print("[DBG] B0  | FB(x)       =", float(FB0.value),
#                 "||gB||    =", float(gB0n.value),
#                 "max|gB|   =", float(np.max(np.abs(gB0.value))),
#                 "step      =", float(sB.value),
#                 "||x||inf  =", _inf_norm(x_cand.value),
#                 "||b1||inf =", _inf_norm(b1.value))

#         # Another common overflow point:
#         FB1 = phi_B(b1)
#         print("[DBG] B1  | FB(b1)      =", float(FB1.value))

#         # ===== B: second projection step =====
#         gB1  = csdl.reshape(csdl.derivative(ofs=FB1, wrts=b1), (3,))
#         gB1n = safe_norm2(gB1) + _DEF_EPS
#         b    = b1 - (FB1 * gB1) / (gB1n * gB1n)

#         print("[DBG] B2  | ||gB(b1)||  =", float(gB1n.value),
#                 "max|gB(b1)|=", float(np.max(np.abs(gB1.value))),
#                 "||b||inf   =", _inf_norm(b.value))

#         # ---- gap (optional) ----
#         diff = a - b
#         gap  = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)
#         print("[DBG] GAP | ||a-b||     =", float(gap.value))

#         # ===== residuals =====
#         FA_x = phi_A(x_cand); FB_x = phi_B(x_cand)
#         r_eq = csdl.absolute(FA_x - FB_x)
#         print("[DBG] EQ  | |FA-FB|     =", float(r_eq.value))

#         FA_a = phi_A(a); gA_a = csdl.reshape(csdl.derivative(ofs=FA_a, wrts=a), (3,))
#         FB_b = phi_B(b); gB_b = csdl.reshape(csdl.derivative(ofs=FB_b, wrts=b), (3,))
#         nA   = gA_a / (safe_norm2(gA_a) + _DEF_EPS)
#         nB   = gB_b / (safe_norm2(gB_b) + _DEF_EPS)

#         print("[DBG] NRM | ||gA(a)||   =", float(safe_norm2(gA_a).value),
#                 "||gB(b)|| =", float(safe_norm2(gB_b).value),
#                 "max|gA(a)|=", float(np.max(np.abs(gA_a.value))),
#                 "max|gB(b)|=", float(np.max(np.abs(gB_b.value))))

#         r_dir = 0.5 * safe_norm2(nA + nB)
#         print("[DBG] DIR | r_dir       =", float(r_dir.value))

#         r_eik_A = csdl.absolute(safe_norm2(gA_a) - 1.0)
#         r_eik_B = csdl.absolute(safe_norm2(gB_b) - 1.0)
#         print("[DBG] EIK | A=", float(r_eik_A.value), "B=", float(r_eik_B.value))


#     if return_all:
#         return xK, F_star, a, b
#     return xK, F_star





# # It works?? Substitute lagrange multipliers
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_stationarity",
#     return_all: bool = False,
#     normalize_grads: bool = True,
#     grad_floor: float = 1e-8,        # guard for ||∇φ||
#     reg_pos: float = 1e-6,           # tiny Tikhonov regularization (anchors at seeds)
#     surf_weight: float = 1.0,        # penalty strength pushing a,b to their surfaces
#     beta_scale_with_grad: bool = True # scale penalty by ||∇φ|| if not normalizing
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:

#     # ----- numeric seeds -----
#     try:
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     except AttributeError:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np = x0_np.copy()
#     b0_np = x0_np.copy()

#     # constants / guards
#     _eps = 1e-12

#     # keep copies of the seeds as constants for regularization anchors
#     a0_const = csdl.Variable(value=a0_np)
#     b0_const = csdl.Variable(value=b0_np)

#     ones3 = csdl.Variable(value=np.ones(3))

#     # ----- states (only a and b) -----
#     a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
#     b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

#     # ----- SDF values & (safe) normals -----
#     # A
#     phiA_a = phi_A(a)                                       # scalar
#     gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     if normalize_grads:
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#         betaA   = surf_weight
#     else:
#         # still guard the zero-gradient case
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#         betaA   = surf_weight * (gA_norm if beta_scale_with_grad else 1.0)

#     # B
#     phiB_b = phi_B(b)                                       # scalar
#     gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))
#     if normalize_grads:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#         betaB   = surf_weight
#     else:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#         betaB   = surf_weight * (gB_norm if beta_scale_with_grad else 1.0)

#     # Reshape scalars for vector scaling
#     phiA_as3 = csdl.reshape(phiA_a, (1,)) * ones3
#     phiB_as3 = csdl.reshape(phiB_b, (1,)) * ones3

#     # ----- residuals (penalized stationarity, no λ/μ states) -----
#     # Intuition: original KKT had (a - b) + λ nA = 0, (a - b) - μ nB = 0 with φ_A(a)=0, φ_B(b)=0.
#     # Here we replace λ ≈ -β φ_A(a), μ ≈  β φ_B(b), which pushes a,b to their surfaces and couples them.
#     RS_A = (a - b) + betaA * (phiA_as3 * nA) + reg_pos * (a - a0_const)  # (3,)
#     RS_B = (a - b) - betaB * (phiB_as3 * nB) + reg_pos * (b - b0_const)  # (3,)

#     # ----- Newton solve -----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     # (Optional) If your CSDL build supports it, you can also set a safe max iters, damping, etc.
#     # e.g., solver.options['max_iter'] = 80
#     solver.add_state(a, RS_A, initial_value=a0_np)   # 3 eqs
#     solver.add_state(b, RS_B, initial_value=b0_np)   # 3 eqs
#     solver.run()

#     # ---------- helpers (shape-safe) ----------
#     def _snap_once(p, phi, grad_floor=1e-8):
#         # One Newton projection step: p <- p - φ * n, with unit-normal and floor
#         val = phi(p)                                              # scalar
#         g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
#         n   = g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)
#         return p - csdl.reshape(val, (1,)) * n                    # (3,)

#     def _snap_k(p0, phi, k=2, grad_floor=1e-8):
#         p = p0
#         for _ in range(k):
#             p = _snap_once(p, phi, grad_floor)
#         return p

#     def _unit(v, eps=1e-12):
#         return v / (csdl.sqrt(csdl.vdot(v, v)) + eps)

#     def _normal_at(p, phi, grad_floor=1e-8):
#         val = phi(p)  # scalar (unused; ensures correct graph deps)
#         g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
#         return g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)

#     # ---------- alternating-projection refinement ----------
#     # Idea: land each point on the hemisphere facing the *other* body by projecting
#     # from the other body’s point. Do both orders and keep the better one.

#     # Option A: start from b toward A, then from a toward B (B→A→B)
#     a_A1 = _snap_k(b, phi_A, k=2, grad_floor=grad_floor)         # project B's point onto A
#     b_A1 = _snap_k(a_A1, phi_B, k=2, grad_floor=grad_floor)      # then project that onto B

#     # Option B: start from a toward B, then from b toward A (A→B→A)
#     b_B1 = _snap_k(a, phi_B, k=2, grad_floor=grad_floor)
#     a_B1 = _snap_k(b_B1, phi_A, k=2, grad_floor=grad_floor)

#     # Score both candidates by (i) smaller gap and (ii) better normal alignment
#     def _score_pair(aP, bP, phiA, phiB):
#         t   = _unit(aP - bP)
#         nA  = _normal_at(aP, phiA, grad_floor)
#         nB  = _normal_at(bP, phiB, grad_floor)
#         gap = csdl.sqrt(csdl.vdot(aP - bP, aP - bP) + 1e-12)
#         # alignment: both outward normals should align with chord (>=0, closer to 1 is better)
#         align = 0.5 * (csdl.vdot(nA, t) + csdl.vdot(nB, t))       # scalar
#         return gap, align

#     gap_A, align_A = _score_pair(a_A1, b_A1, phi_A, phi_B)
#     gap_B, align_B = _score_pair(a_B1, b_B1, phi_A, phi_B)

#     # Choose the better pair: prefer smaller gap; break ties by larger alignment
#     # (CSDL has no conditionals; pick both and *use the one you return* in Python flow)
#     use_A = True  # Python-level choice is fine since you're building the graph once per call
#     # You can fetch initial seeds’ numpy to decide, but simplest is:
#     #   if you want fully CSDL, keep both and return both to caller to choose.
#     try:
#         # If shapes were numeric Variables, take a quick numpy read for the decision:
#         use_A = (float(gap_A.value) < float(gap_B.value)) or \
#                 (abs(float(gap_A.value) - float(gap_B.value)) < 1e-9 and float(align_A.value) >= float(align_B.value))
#     except Exception:
#         # Fallback: prefer A by default
#         use_A = True

#     a_ref = a_A1 if use_A else a_B1
#     b_ref = b_A1 if use_A else b_B1

#     # # Optional tiny clean-up snap to ensure |φ| ~ 0 after the alternations
#     # a_ref = _snap_once(a_ref, phi_A, grad_floor)
#     # b_ref = _snap_once(b_ref, phi_B, grad_floor)

#     # ---------- replace outputs ----------
#     a = a_ref
#     b = b_ref
#     diff = a - b
#     gap  = csdl.sqrt(csdl.vdot(diff, diff) + 1e-12)
#     x_mid  = 0.5 * (a + b)
#     F_star = csdl.maximum(phi_A(x_mid), phi_B(x_mid), rho=20.0)- 0.07

#     if return_all:
#         return x_mid, F_star, a, b, gap
#     return gap



#IDK ANYMORE KKT AGAIN
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_stationarity",
#     return_all: bool = False,
#     normalize_grads: bool = True,
#     grad_floor: float = 1e-8,        # guard for ||∇φ||
#     reg_pos: float = 1e-6            # tiny Tikhonov regularization in stationarity eqs
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:

#     # ----- numeric seeds -----
#     try:
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     except AttributeError:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np   = x0_np.copy()
#     b0_np   = x0_np.copy()
#     lam0_np = np.array([0.0], dtype=float)
#     mu0_np  = np.array([0.0], dtype=float)

#     # constants / guards
#     _eps = 1e-12 if "_DEF_EPS" in globals() else 1e-12

#     # keep copies of the seeds as constants for regularization anchors
#     a0_const = csdl.Variable(value=a0_np)
#     b0_const = csdl.Variable(value=b0_np)

#     # ----- states -----
#     a   = csdl.Variable(name=f"{newton_name}:a",   shape=(3,), value=a0_np)
#     b   = csdl.Variable(name=f"{newton_name}:b",   shape=(3,), value=b0_np)
#     lam = csdl.Variable(name=f"{newton_name}:lam", shape=(1,), value=lam0_np)
#     mu  = csdl.Variable(name=f"{newton_name}:mu",  shape=(1,), value=mu0_np)

#     # ----- SDF values & (safe) normals -----
#     # A
#     phiA_a = phi_A(a)                                        # scalar
#     gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     if normalize_grads:
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#     else:
#         # still guard the zero-gradient case
#         scale   = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / scale

#     # B
#     phiB_b = phi_B(b)                                        # scalar
#     gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))
#     if normalize_grads:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#     else:
#         scale   = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / scale

#     # broadcast λ, μ to 3-vectors
#     ones3 = csdl.Variable(value=np.ones(3))
#     lam3  = csdl.reshape(csdl.tensordot(lam, ones3, axes=None), (3,))
#     mu3   = csdl.reshape(csdl.tensordot(mu,  ones3, axes=None), (3,))

#     # ----- residuals -----
#     # Surface constraints
#     RA = csdl.reshape(phiA_a, (1,))                    # φ_A(a) = 0
#     RB = csdl.reshape(phiB_b, (1,))                    # φ_B(b) = 0

#     # Stationarity with tiny Tikhonov (anchors at seeds)
#     #   (a - b) + λ nA + reg*(a - a0) = 0
#     #   (a - b) - μ nB + reg*(b - b0) = 0
#     RS_A = (a - b) + lam3 * nA + reg_pos * (a - a0_const)   # (3,)
#     RS_B = (a - b) - mu3  * nB + reg_pos * (b - b0_const)   # (3,)

#     # ----- Newton solve -----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a,   RS_A, initial_value=a0_np)   # 3 eqs
#     solver.add_state(b,   RS_B, initial_value=b0_np)   # 3 eqs
#     solver.add_state(lam, RA,   initial_value=lam0_np) # 1 eq
#     solver.add_state(mu,  RB,   initial_value=mu0_np)  # 1 eq
#     solver.run()

#     # ----- outputs -----
#     diff = a - b
#     gap  = csdl.sqrt(csdl.vdot(diff, diff) + _eps)

#     # mid-point & smooth max (kept for API compatibility)
#     x_mid  = 0.5 * (a + b)
#     F_star = csdl.maximum(phi_A(x_mid), phi_B(x_mid), rho=20.0)

#     if return_all:
#         return x_mid, F_star, a, b, gap
#     return gap











# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     gamma: float = 50.0,                 # penalty strength (try 1–20 depending on units)
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_penalty",
#     return_all: bool = False,
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Fully differentiable collision check via the penalized closest-pair stationarity:
#         (a - b) + gamma * phi_A(a) * gA_hat = 0
#         (b - a) + gamma * phi_B(b) * gB_hat = 0
#     Returns pair_gap by default; with return_all=True returns (m, F_star, a, b, pair_gap).
#     """

#     # ---- numeric seeds (leaf states) ----
#     m0 = x0.value.reshape(3,) if isinstance(x0, csdl.Variable) else np.asarray(x0, float).reshape(3,)
#     a0_np = m0.copy()
#     b0_np = m0.copy()

#     a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
#     b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

#     # φ and gradients
#     phiA = phi_A(a)  # scalar
#     phiB = phi_B(b)  # scalar

#     gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))

#     # Stationarity residuals (6 equations,3 each)
#     R_a = (a - b) + gamma * phiA * gA   # (3,)
#     R_b = (b - a) + gamma * phiB * gB   # (3,)

#     # Newton solve 
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a, R_a, initial_value=a0_np)
#     solver.add_state(b, R_b, initial_value=b0_np)
#     solver.run()

#     m = 0.5 * (a + b)

#     F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.0507

#     if return_all:
#         # Additional Outputs
#         diff = a - b
#         pair_gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#         return  m, F_star, a, b, pair_gap
#     return F_star


# =================== KKT ATTEMPT ====================
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     gamma: float = 50.0,                 # penalty strength (try 1–20 depending on units)
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_penalty",
#     return_all: bool = False,
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Fully differentiable collision check via the penalized closest-pair stationarity:
#         (a - b) + gamma * phi_A(a) * gA_hat = 0
#         (b - a) + gamma * phi_B(b) * gB_hat = 0
#     Returns pair_gap by default; with return_all=True returns (m, F_star, a, b, pair_gap).
#     """

#     # ---- numeric seeds (leaf states) ----
#     m0 = x0.value.reshape(3,) if isinstance(x0, csdl.Variable) else np.asarray(x0, float).reshape(3,)
#     a0_np = m0.copy()
#     b0_np = m0.copy()

#     # Unknowns
#     a = csdl.Variable((3,), value=a0_np)
#     b = csdl.Variable((3,), value=b0_np)
#     lamA = csdl.Variable(value=0.0)
#     lamB = csdl.Variable(value=0.0)

#     phiA = phi_A(a); gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
#     phiB = phi_B(b); gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))

#     R_a = (a - b) + lamA * gA
#     R_b = (b - a) + lamB * gB
#     C_A = phiA               # equality constraint
#     C_B = phiB

#     solver = csdl.nonlinear_solvers.Newton("closest_pair_kkt", tolerance=1e-8)
#     solver.add_state(a, R_a)
#     solver.add_state(b, R_b)
#     solver.add_state(lamA, C_A)  # enforces phi_A(a)=0
#     solver.add_state(lamB, C_B)  # enforces phi_B(b)=0
#     solver.run()


#     m = 0.5 * (a + b)

#     F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.0507

#     if return_all:
#         # Additional Outputs
#         diff = a - b
#         pair_gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#         return  m, F_star, a, b, pair_gap
#     return F_star

# =================== END KKT ATTEMPT ====================






















# Leftover helper for below
# def _project_from_x(x: csdl.Variable, phi_x: csdl.Variable, grad_x: csdl.Variable,
#                     eps: float = _DEF_EPS) -> csdl.Variable:
#     """First-order projection of x onto the zero level set of an SDF.
#     a = x - phi(x) * grad/||grad||^2
#     """
#     denom = csdl.sum(grad_x * grad_x) + eps
#     return x - (phi_x * grad_x) / denom


# Gradient of F formula (issues with deep collisions)
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check",
#     return_all: bool = False,
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Perform collision check between two SDFs.

#     Parameters
#     ----------
#     phi_A, phi_B : callable
#         Functions mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
#     x0 : np.ndarray | csdl.Variable, shape (3,)
#         Initial seed for the Newton solve. May be a NumPy array or a CSDL Variable.
#     newton_tol : float
#         Convergence tolerance.
#     newton_name : str
#         Name for the CSDL Newton solver instance.
#     return_all : bool
#         If True, return tuple (x_star, F_star, a, b, pair_gap).
#         Else return just pair_gap.
#     """
#     if isinstance(x0, csdl.Variable):
#         x = x0
#         use_initial_value = False
#         init_val = None
#     else:
#         x = csdl.Variable(name=f"{newton_name}:x", shape=(3,), value=np.asarray(x0, dtype=float))
#         use_initial_value = True
#         init_val = np.asarray(x0, dtype=float)

#     phiA_x = phi_A(x)
#     phiB_x = phi_B(x)
#     F = csdl.maximum(phiA_x, phiB_x)

#     gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))

#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     if use_initial_value:
#         solver.add_state(x, gF, initial_value=init_val)
#     else:
#         solver.add_state(x, gF)
#     solver.run()

#     phiA_x = phi_A(x)
#     phiB_x = phi_B(x)
#     gA = csdl.reshape(csdl.derivative(ofs=phiA_x, wrts=x), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB_x, wrts=x), (3,))

#     a = _project_from_x(x, phiA_x, gA)
#     b = _project_from_x(x, phiB_x, gB)
#     pair_gap = csdl.norm(a - b)

#     if return_all:
#         return x, F, a, b, pair_gap
#     return pair_gap

# __all__ = ["collision_check"]


# KKT approach (issues with Nan/Inf/Singular Matrix)
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_kkt",
#     return_all: bool = False,
#     normalize_grads: bool = False,
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Closest-pair collision check via KKT system:

#         a - x - λ_A ∇φ_A(a) = 0
#         b - x - λ_B ∇φ_B(b) = 0
#         φ_A(a) = 0
#         φ_B(b) = 0
#         x - (a + b)/2 = 0

#     Parameters
#     ----------
#     phi_A, phi_B : callable
#         SDFs mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
#     x0 : np.ndarray | csdl.Variable
#         Numeric seed for the midpoint state x (if csdl.Variable, its .value is used).
#     newton_tol : float
#         Newton tolerance.
#     newton_name : str
#         Name for the Newton solver instance.
#     return_all : bool
#         If True, returns (x_star, F_star, a_star, b_star, pair_gap).
#         Else returns just pair_gap.
#     normalize_grads : bool
#         If True, uses unit normals g/||g|| in the KKT equations.

#     Returns
#     -------
#     tuple or csdl.Variable
#         If return_all=True: (x, F_star, a, b, pair_gap).
#         Else: pair_gap.
#     """
#     # ---- numeric seeds (keep states as leaf variables) ----
#     if isinstance(x0, csdl.Variable):
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     else:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np   = x0_np.copy()
#     b0_np   = x0_np.copy()
#     lamA0_np = np.array([0.0], dtype=float)
#     lamB0_np = np.array([0.0], dtype=float)

#     # ---- states ----
#     a    = csdl.Variable(name=f"{newton_name}:a",    shape=(3,), value=a0_np)
#     lamA = csdl.Variable(name=f"{newton_name}:lamA", shape=(1,), value=lamA0_np)
#     b    = csdl.Variable(name=f"{newton_name}:b",    shape=(3,), value=b0_np)
#     lamB = csdl.Variable(name=f"{newton_name}:lamB", shape=(1,), value=lamB0_np)
#     x    = csdl.Variable(name=f"{newton_name}:x",    shape=(3,), value=x0_np)

#     # ---- values & gradients on each surface point ----
#     phiA_a = phi_A(a)                              # scalar
#     phiB_b = phi_B(b)                              # scalar
#     gA = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))

#     if normalize_grads:
#         gA = gA / csdl.sqrt(csdl.vdot(gA, gA) + _DEF_EPS)
#         gB = gB / csdl.sqrt(csdl.vdot(gB, gB) + _DEF_EPS)

#     # broadcast λ to 3-vector
#     ones3 = csdl.Variable(value=np.ones(3))
#     lamA3 = csdl.reshape(csdl.tensordot(lamA, ones3, axes=None), (3,))
#     lamB3 = csdl.reshape(csdl.tensordot(lamB, ones3, axes=None), (3,))

#     # ---- residuals ----
#     RA1 = a - x - lamA3 * gA              # (3,)
#     RA2 = csdl.reshape(phiA_a, (1,))      # (1,)
#     RB1 = b - x - lamB3 * gB              # (3,)
#     RB2 = csdl.reshape(phiB_b, (1,))      # (1,)
#     RX  = x - 0.5*(a + b)                 # (3,)

#     # ---- Newton solve ----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a,    RA1, initial_value=a0_np)
#     solver.add_state(lamA, RA2, initial_value=lamA0_np)
#     solver.add_state(b,    RB1, initial_value=b0_np)
#     solver.add_state(lamB, RB2, initial_value=lamB0_np)
#     solver.add_state(x,    RX,  initial_value=x0_np)
#     solver.run()

#     # ---- outputs ----
#     # pair gap (guarded)
#     diff = a - b
#     gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#     # F_star for compatibility with your previous return tuple
#     F_star = csdl.maximum(phi_A(x), phi_B(x), rho=20.0)

#     if return_all:
#         return x, F_star, a, b, gap
#     return gap


# __all__ = [
#     "collision_check_kkt",
# ]

# # ------------------------------------------------------------
# # Two-phase collision solve with feasibility weighting
# # Phase 1: LM projection to each surface (gets |phi| ~ 0)
# # Phase 2: KKT LM with weighted feasibility (robust from far seeds)
# # ------------------------------------------------------------

# def collision_check_kkt_LM(
#     phi_A, phi_B, x0,
#     *,
#     # Phase-1 (projection) LM settings
#     proj_max_iter: int = 40,
#     proj_tol: float = 1e-12,
#     # Phase-2 (KKT) LM settings
#     kkt_max_iter: int = 80,
#     kkt_tol: float = 1e-12,
#     lm_mu0: float = 1e-2,
#     lm_mu_inc: float = 10.0,
#     lm_mu_dec: float = 1/3,
#     lm_eta: float = 1e-3,             # gain ratio acceptance threshold
#     # Feasibility weighting schedule (for Phase-2)
#     feas_weight_init: float = 50.0,    # start heavy so φ-terms dominate when far
#     feas_weight_decay: float = 0.5,    # multiply weight by this each schedule step
#     feas_taper_threshold: float = 1e-3,# once max(|φ|) below this, set weight=1
#     feas_schedule_steps: int = 3,      # how many weighted passes before w=1
#     # Misc
#     return_all: bool = False
# ):
#     """
#     Returns: gap (default)
#              or (x_mid, a, b, gap) if return_all=True
#     """

#     # ---------- utils to extract numpy ----------
#     def _to_np(x, shape):
#         try:
#             return np.asarray(x.value, float).reshape(*shape)
#         except Exception:
#             return np.asarray(x, float).reshape(*shape)

#     # ---------- Phase-1: LM projection to a surface ----------
#     def _lm_project(phi, p0_np):
#         mu = float(lm_mu0)
#         p = p0_np.copy()

#         def FJ(p_np):
#             p_v = csdl.Variable(value=p_np.reshape(3,))
#             val = phi(p_v)                                        # scalar
#             g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p_v),(3,))
#             F   = np.array([float(val.value)], float)             # (1,)
#             J   = np.zeros((1,3), float); J[0,:] = _to_np(g,(3,))
#             return F, J

#         F, J = FJ(p); f2 = float(F @ F)
#         for _ in range(proj_max_iter):
#             JTJ = J.T @ J
#             g = J.T @ F
#             try:
#                 dp = -np.linalg.solve(JTJ + mu*np.eye(3), g).reshape(3,)
#             except np.linalg.LinAlgError:
#                 mu *= lm_mu_inc; continue
#             p_trial = p + dp
#             F_trial, J_trial = FJ(p_trial); f2_trial = float(F_trial @ F_trial)

#             # predicted reduction (quadratic model on 1D residual)
#             pred_red = - (dp @ g).item() - 0.5 * (dp @ (JTJ @ dp)).item()
#             rho = (f2 - f2_trial) / max(pred_red, 1e-16)

#             if rho > lm_eta and f2_trial < f2:
#                 p, F, J, f2 = p_trial, F_trial, J_trial, f2_trial
#                 mu = max(mu * lm_mu_dec, 1e-15)
#                 if f2 < proj_tol: break
#             else:
#                 mu *= lm_mu_inc
#         return p

#     # ---------- Phase-2: weighted KKT LM (one run) ----------
#     # Unknowns: a(3), b(3), lamA(1), lamB(1) -> x in R^8
#     def _kkt_LM(a_seed_np, b_seed_np, w_phi: float, max_iter: int, tol: float):
#         def pack(a, b, lamA, lamB):
#             x = np.zeros(8, float)
#             x[0:3] = a; x[3:6] = b; x[6] = lamA[0]; x[7] = lamB[0]
#             return x
#         def unpack(x):
#             return x[0:3], x[3:6], np.array([x[6]]), np.array([x[7]])

#         mu = float(lm_mu0)
#         x = pack(a_seed_np, b_seed_np, np.array([0.0]), np.array([0.0]))

#         def FJ(x_vec):
#             a_np, b_np, lamA_np, lamB_np = unpack(x_vec)

#             a_v = csdl.Variable(value=a_np.reshape(3,))
#             b_v = csdl.Variable(value=b_np.reshape(3,))
#             lamA_v = csdl.Variable(value=lamA_np.reshape(1,))
#             lamB_v = csdl.Variable(value=lamB_np.reshape(1,))

#             # SDFs and gradients
#             phiA_a = phi_A(a_v); gA = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a_v), (3,))
#             phiB_b = phi_B(b_v); gB = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b_v), (3,))

#             diff = a_v - b_v

#             # Residuals:
#             # R1 = (a-b) + lamA * grad(phi_A)(a)    (3)
#             # R2 = -(a-b) + lamB * grad(phi_B)(b)   (3)
#             # R3 = w_phi * phi_A(a)                 (1)
#             # R4 = w_phi * phi_B(b)                 (1)
#             R1 = diff + csdl.reshape(lamA_v,(1,)) * gA
#             R2 = -diff + csdl.reshape(lamB_v,(1,)) * gB
#             R3 = w_phi * phiA_a
#             R4 = w_phi * phiB_b

#             F = np.zeros(8, float)
#             F[0:3] = _to_np(R1,(3,))
#             F[3:6] = _to_np(R2,(3,))
#             F[6]   = float(R3.value)
#             F[7]   = float(R4.value)

#             # Jacobian via AD row-by-row
#             J = np.zeros((8,8), float)

#             def fill_row(i, r_scalar):
#                 da = csdl.derivative(ofs=r_scalar, wrts=a_v)  # (3,)
#                 db = csdl.derivative(ofs=r_scalar, wrts=b_v)  # (3,)
#                 dA = csdl.derivative(ofs=r_scalar, wrts=lamA_v) # (1,)
#                 dB = csdl.derivative(ofs=r_scalar, wrts=lamB_v) # (1,)
#                 J[i,0:3] = _to_np(da,(3,))
#                 J[i,3:6] = _to_np(db,(3,))
#                 J[i,6]   = float(dA.value)
#                 J[i,7]   = float(dB.value)

#             # R1 rows
#             fill_row(0, R1[0]); fill_row(1, R1[1]); fill_row(2, R1[2])
#             # R2 rows
#             fill_row(3, R2[0]); fill_row(4, R2[1]); fill_row(5, R2[2])
#             # R3, R4 rows
#             fill_row(6, R3); fill_row(7, R4)

#             # Also return raw φ for taper decisions
#             return F, J, float(phiA_a.value), float(phiB_b.value)

#         F, J, phiA_val, phiB_val = FJ(x); f2 = float(F @ F)
#         for _ in range(max_iter):
#             JTJ = J.T @ J
#             g = J.T @ F
#             try:
#                 dx = -np.linalg.solve(JTJ + mu*np.eye(8), g)
#             except np.linalg.LinAlgError:
#                 mu *= lm_mu_inc; continue

#             x_trial = x + dx
#             F_trial, J_trial, phiA_t, phiB_t = FJ(x_trial)
#             f2_trial = float(F_trial @ F_trial)

#             # Predicted reduction (Gauss–Newton model)
#             pred_red = - (dx @ g) - 0.5 * (dx @ (JTJ @ dx))
#             rho = (f2 - f2_trial) / max(pred_red, 1e-16)

#             if rho > lm_eta and f2_trial < f2:
#                 x, F, J, f2 = x_trial, F_trial, J_trial, f2_trial
#                 phiA_val, phiB_val = phiA_t, phiB_t
#                 mu = max(mu * lm_mu_dec, 1e-15)
#                 if f2 < kkt_tol:
#                     break
#             else:
#                 mu *= lm_mu_inc

#         a_np, b_np, lamA_np, lamB_np = unpack(x)
#         return a_np, b_np, lamA_np, lamB_np, f2, max(abs(phiA_val), abs(phiB_val))

#     # ---------- Build Phase-1 seeds (two orders), run Phase-2 with schedule ----------
#     x0_np = _to_np(x0, (3,))

#     # Order A→B
#     a_seed_A = _lm_project(phi_A, x0_np)
#     b_seed_A = _lm_project(phi_B, a_seed_A)

#     # Order B→A
#     b_seed_B = _lm_project(phi_B, x0_np)
#     a_seed_B = _lm_project(phi_A, b_seed_B)

#     def run_schedule(a0_np, b0_np):
#         a_np, b_np = a0_np.copy(), b0_np.copy()
#         # Feasibility-weighted passes
#         w = max(1.0, float(feas_weight_init))
#         for s in range(max(1, feas_schedule_steps)):
#             a_np, b_np, lamA_np, lamB_np, f2, feas = _kkt_LM(
#                 a_np, b_np, w_phi=w, max_iter=kkt_max_iter, tol=kkt_tol
#             )
#             # Taper weight if feasible enough
#             if feas <= feas_taper_threshold:
#                 w = 1.0
#                 break
#             w = max(1.0, w * float(feas_weight_decay))
#         # Final refinement at weight=1 (ensures unbiased stationarity)
#         a_np, b_np, lamA_np, lamB_np, f2, feas = _kkt_LM(
#             a_np, b_np, w_phi=1.0, max_iter=kkt_max_iter, tol=kkt_tol
#         )
#         return a_np, b_np, f2, feas

#     aA_np, bA_np, f2A, feasA = run_schedule(a_seed_A, b_seed_A)
#     aB_np, bB_np, f2B, feasB = run_schedule(a_seed_B, b_seed_B)

#     # Pick the better candidate (prefer smaller gap; tie-break by feasibility & f2)
#     def gap_of(a_np, b_np):
#         diff = a_np - b_np
#         return float(np.sqrt(np.dot(diff, diff)))

#     gapA = gap_of(aA_np, bA_np)
#     gapB = gap_of(aB_np, bB_np)

#     if (gapA < gapB) or (abs(gapA-gapB) < 1e-12 and (feasA, f2A) <= (feasB, f2B)):
#         a_best, b_best, gap_best = aA_np, bA_np, gapA
#     else:
#         a_best, b_best, gap_best = aB_np, bB_np, gapB

#     # ---- Pack outputs as CSDL Variables (for downstream graph use) ----
#     a_v = csdl.Variable(value=a_best.reshape(3,))
#     b_v = csdl.Variable(value=b_best.reshape(3,))
#     diff_v = a_v - b_v
#     gap_v = csdl.sqrt(csdl.vdot(diff_v, diff_v) + 1e-16)
#     x_mid = 0.5 * (a_v + b_v)

#     if return_all:
#         return x_mid, a_v, b_v, gap_v
#     return gap_v







# _DEF_EPS = 1e-12

# ================ HYBRID METHOD ================
# def _as_array(x, shape=None):
#     arr = np.asarray(getattr(x, 'value', x), float)
#     return arr if shape is None else arr.reshape(shape)

# def _quat_to_R_wb(q):
#     """Rotation (WORLD ← BODY) from quaternion q=[w,x,y,z] (numpy)."""
#     w, x, y, z = q
#     n = np.linalg.norm(q)
#     if n == 0:
#         return np.eye(3)
#     w, x, y, z = q / n
#     return np.array([
#         [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
#         [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
#         [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
#     ])

# def quat_to_rm_csdl(q: csdl.Variable) -> csdl.Variable:
#     """Quaternion [w,x,y,z] -> 3x3 rotation (CSDL). Normalizes internally."""
#     qw, qx, qy, qz = q[0], q[1], q[2], q[3]
#     nrm = csdl.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + _DEF_EPS
#     qw, qx, qy, qz = qw/nrm, qx/nrm, qy/nrm, qz/nrm
#     r00 = 1.0 - 2.0*(qy*qy + qz*qz)
#     r01 = 2.0*(qx*qy - qw*qz)
#     r02 = 2.0*(qx*qz + qw*qy)
#     r10 = 2.0*(qx*qy + qw*qz)
#     r11 = 1.0 - 2.0*(qx*qx + qz*qz)
#     r12 = 2.0*(qy*qz - qw*qx)
#     r20 = 2.0*(qx*qz - qw*qy)
#     r21 = 2.0*(qy*qz + qw*qx)
#     r22 = 1.0 - 2.0*(qx*qx + qy*qy)
#     R_flat = csdl.vstack((r00, r01, r02, r10, r11, r12, r20, r21, r22))
#     return csdl.reshape(R_flat, (3, 3))

# # ---------- main entry ----------
# def collision_check(
#     phi_A_body_cm: Callable[[csdl.Variable], csdl.Variable],
#     phi_B_world_m: Callable[[csdl.Variable], csdl.Variable],
#     pos_w_m: csdl.Variable,
#     quat_wxyz: csdl.Variable,
#     *,
#     b_seed_center_w_m: np.ndarray,    # e.g., gate center
#     max_refine: int = 8,
#     enable_phase2: bool = True,
#     plot_octree: bool = True,
#     plot_kkt: bool = True,
#     gamma: float = 50.0,
#     newton_tol: float = 1e-8,
#     recorder: Optional[csdl.Recorder] = None
# ) -> Dict[str, object]:
#     """
#     Returns a dict with:
#       - keptA, halfA, intervalsA, keptB, halfB, intervalsB (by level)
#       - mnn_pairs (final), C_A_top (1,3), C_B_top (1,3)
#       - KKT: m, F_star, a, b, pair_gap  (csdl.Variables)
#     """

#     # =============== OFFLINE: dual-sided octree + MNN (pause main recorder) ===============
#     if recorder is not None:
#         recorder.stop()
#     dummy = csdl.Recorder(inline=True); dummy.start()

#     # pose (numpy)
#     p_w  = _as_array(pos_w_m, (3,))
#     R_wb = _quat_to_R_wb(_as_array(quat_wxyz, (4,)))

#     # numeric wrappers
#     def phi_A_numeric_world(P_w_m: np.ndarray) -> np.ndarray:
#         P_w_m = np.asarray(P_w_m, float).reshape(-1, 3)
#         P_b_m  = (P_w_m - p_w) @ R_wb.T
#         P_b_cm = 100.0 * P_b_m
#         vals_cm = phi_A_body_cm(csdl.Variable(value=P_b_cm)).value
#         return 0.01 * np.asarray(vals_cm).reshape(-1)

#     def phi_B_numeric_world(P_w_m: np.ndarray) -> np.ndarray:
#         P_w_m = np.asarray(P_w_m, float).reshape(-1, 3)
#         vals = phi_B_world_m(csdl.Variable(value=P_w_m)).value
#         return np.asarray(vals).reshape(-1)

#     # voxel structure
#     class Voxels:
#         def __init__(self, nodes: np.ndarray, L: int, bounds):
#             self.nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
#             self.L = int(L)
#             self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = map(float, bounds)
#         @classmethod
#         def root(cls, bounds):
#             return cls(nodes=np.array([[0,0,0]]), L=0, bounds=bounds)
#         def _sizes(self):
#             W = self.xmax - self.xmin; H = self.ymax - self.ymin; D = self.zmax - self.zmin
#             n = 1 << self.L
#             dx, dy, dz = W/n, H/n, D/n
#             hx, hy, hz = 0.5*dx, 0.5*dy, 0.5*dz
#             return dx, dy, dz, hx, hy, hz
#         def centers(self):
#             dx, dy, dz, _, _, _ = self._sizes()
#             i = self.nodes[:,0].astype(float); j = self.nodes[:,1].astype(float); k = self.nodes[:,2].astype(float)
#             cx = self.xmin + (i + 0.5) * dx
#             cy = self.ymin + (j + 0.5) * dy
#             cz = self.zmin + (k + 0.5) * dz
#             return np.column_stack([cx, cy, cz])
#         def half_sizes(self):
#             _, _, _, hx, hy, hz = self._sizes()
#             return float(hx), float(hy), float(hz)
#         def radius(self):
#             hx, hy, hz = self.half_sizes()
#             return float(np.sqrt(hx*hx + hy*hy + hz*hz))
#         def subdivide(self):
#             i = self.nodes[:,0]; j = self.nodes[:,1]; k = self.nodes[:,2]
#             kids = np.array([
#                 np.stack(((i<<1)|0, (j<<1)|0, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|0, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|0, (j<<1)|1, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|1, (k<<1)|0), axis=1),
#                 np.stack(((i<<1)|0, (j<<1)|0, (k<<1)|1), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|0, (k<<1)|1), axis=1),
#                 np.stack(((i<<1)|0, (j<<1)|1, (k<<1)|1), axis=1),
#                 np.stack(((i<<1)|1, (j<<1)|1, (k<<1)|1), axis=1),
#             ]).reshape(-1, 3)
#             return Voxels(kids, self.L+1, (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax))
#         def restrict(self, mask):
#             self.nodes = self.nodes[np.asarray(mask, bool)]

#     # robust autosize (minimal tweak): ensure half ≥ |φ(seed)| + margin
#     def _autosize_root_bbox_from(phi_fn_numeric, seed_center, hit_thresh=0.01, ray_max=0.80, margin=0.05):
#         dirs = []
#         s = 1/np.sqrt(2); t = 1/np.sqrt(3)
#         axes  = [[+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]]
#         edges = [[+s,+s,0],[+s,-s,0],[-s,+s,0],[-s,-s,0],
#                  [+s,0,+s],[+s,0,-s],[-s,0,+s],[-s,0,-s],
#                  [0,+s,+s],[0,+s,-s],[0,-s,+s],[0,-s,-s]]
#         corners = [[+t,+t,+t],[+t,+t,-t],[+t,-t,+t],[+t,-t,-t],
#                    [-t,+t,+t],[-t,+t,-t],[-t,-t,+t],[-t,-t,-t]]
#         for d in axes + edges + corners:
#             v = np.asarray(d, float); v /= np.linalg.norm(v); dirs.append(v)

#         radii = np.linspace(0.05, ray_max, 32)
#         max_hit = 0.25
#         for d in dirs:
#             points = seed_center[None,:] + radii[:,None]*d[None,:]
#             phis = phi_fn_numeric(points)
#             if (phis > hit_thresh).any():
#                 idx = np.argmax(phis > hit_thresh)
#                 hit_r = radii[min(idx+1, len(radii)-1)]
#             else:
#                 hit_r = radii[-1]
#             max_hit = max(max_hit, float(hit_r))

#         half = max_hit + margin
#         phi0 = float(phi_fn_numeric(np.asarray(seed_center, float).reshape(1, -1))[0])
#         half = max(half, abs(phi0) + margin)

#         return (seed_center[0]-half, seed_center[0]+half,
#                 seed_center[1]-half, seed_center[1]+half,
#                 seed_center[2]-half, seed_center[2]+half)

#     # A-centric (seed at drone) and B-centric (seed at gate center)
#     rootA = _autosize_root_bbox_from(phi_A_numeric_world, p_w)
#     b_seed = np.asarray(b_seed_center_w_m, float).reshape(3,)
#     rootB = _autosize_root_bbox_from(phi_B_numeric_world, b_seed)

#     # one refinement step (A side)
#     def _refine_once_A(vox: "Voxels"):
#         vox = vox.subdivide()
#         r   = vox.radius()
#         C   = vox.centers()
#         keep1 = (phi_A_numeric_world(C) <= r)                    # Phase-1(A)
#         vox.restrict(keep1)
#         interval = (np.nan, np.nan)
#         if enable_phase2 and len(vox.nodes) > 0:
#             C2   = vox.centers()
#             phiB = phi_B_numeric_world(C2)
#             min_u = np.min(phiB) + r                              # “old” +r then +r → +2r window
#             keep2 = (phiB <= (min_u + r))                         # Phase-2(A)
#             vox.restrict(keep2)
#             interval = (min_u - 2.0*r, min_u)
#         return vox, interval

#     # one refinement step (B side)
#     def _refine_once_B(vox: "Voxels"):
#         vox = vox.subdivide()
#         r   = vox.radius()
#         C   = vox.centers()
#         keep1 = (phi_B_numeric_world(C) <= r)                    # Phase-1(B)
#         vox.restrict(keep1)
#         interval = (np.nan, np.nan)
#         if enable_phase2 and len(vox.nodes) > 0:
#             C2   = vox.centers()
#             phiA = phi_A_numeric_world(C2)
#             min_u = np.min(phiA) + r
#             keep2 = (phiA <= (min_u + r))                         # Phase-2(B)
#             vox.restrict(keep2)
#             interval = (min_u - 2.0*r, min_u)
#         return vox, interval

#     # run refinements
#     voxA = Voxels.root(rootA)
#     keptA, intervalsA, halfA = {0: voxA.centers().copy()}, {0: (np.nan, np.nan)}, {0: voxA.half_sizes()}
#     for L in range(max_refine):
#         voxA, interA = _refine_once_A(voxA)
#         keptA[L+1]      = voxA.centers().copy()
#         intervalsA[L+1] = interA
#         halfA[L+1]      = voxA.half_sizes()

#     voxB = Voxels.root(rootB)
#     keptB, intervalsB, halfB = {0: voxB.centers().copy()}, {0: (np.nan, np.nan)}, {0: voxB.half_sizes()}
#     for L in range(max_refine):
#         voxB, interB = _refine_once_B(voxB)
#         keptB[L+1]      = voxB.centers().copy()
#         intervalsB[L+1] = interB
#         halfB[L+1]      = voxB.half_sizes()

#     # final clouds
#     C_A_final = keptA[max_refine]
#     C_B_final = keptB[max_refine]
#     hxA, hyA, hzA = halfA[max_refine]
#     hxB, hyB, hzB = halfB[max_refine]
#     rA = float(np.linalg.norm([hxA, hyA, hzA]))
#     rB = float(np.linalg.norm([hxB, hyB, hzB]))

#     # MNN pairs
#     mnn_pairs: List[Tuple[int,int]] = []
#     if len(C_A_final) > 0 and len(C_B_final) > 0:
#         d2 = np.sum((C_A_final[:,None,:] - C_B_final[None,:,:])**2, axis=2)
#         nnB_of_A = np.argmin(d2, axis=1)
#         nnA_of_B = np.argmin(d2, axis=0)
#         for iA, jB in enumerate(nnB_of_A):
#             if nnA_of_B[jB] == iA:
#                 mnn_pairs.append((iA, jB))

#     # rank and take top-1
#     def _pair_score(iA, jB):
#         a = phi_A_numeric_world(C_A_final[iA:iA+1])[0]
#         b = phi_B_numeric_world(C_B_final[jB:jB+1])[0]
#         dist = float(np.linalg.norm(C_A_final[iA] - C_B_final[jB]))
#         return np.sqrt(a*a + b*b) + dist - (rA + rB)

#     C_A_top = np.zeros((0,3)); C_B_top = np.zeros((0,3)); top_pair = None
#     if mnn_pairs:
#         scores = np.array([_pair_score(iA, jB) for (iA, jB) in mnn_pairs], float)
#         order  = np.argsort(scores)
#         idx0   = int(order[0])
#         top_pair = mnn_pairs[idx0]
#         C_A_top = C_A_final[[top_pair[0]]]
#         C_B_top = C_B_final[[top_pair[1]]]

#     # restore main recorder for online KKT
#     dummy.stop()
#     if recorder is not None:
#         recorder.start()

#     # optional OCTREE plotting
#     if plot_octree:
#         try:
#             import pyvista as pv
#             # bounds union helper
#             def _cloud_bounds(C, h):
#                 if len(C)==0: return None
#                 hx, hy, hz = h
#                 H = np.tile(np.array([hx, hy, hz], float), (len(C), 1))
#                 mins = C - H; maxs = C + H
#                 return mins.min(axis=0), maxs.max(axis=0)

#             bA = _cloud_bounds(keptA[max_refine], halfA[max_refine])
#             bB = _cloud_bounds(keptB[max_refine], halfB[max_refine])
#             if bA and bB:
#                 mins = np.minimum(bA[0], bB[0]); maxs = np.maximum(bA[1], bB[1])
#             elif bA: mins, maxs = bA
#             elif bB: mins, maxs = bB
#             else:
#                 mins = np.array([rootA[0], rootA[2], rootA[4]])
#                 maxs = np.array([rootA[1], rootA[3], rootA[5]])
#             pad = 0.35
#             xmin, ymin, zmin = mins - pad; xmax, ymax, zmax = maxs + pad

#             Nx = Ny = Nz = 96
#             x = np.linspace(xmin, xmax, Nx)
#             y = np.linspace(ymin, ymax, Ny)
#             z = np.linspace(zmin, zmax, Nz)
#             X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
#             P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
#             phiA = phi_A_numeric_world(P).reshape(X.shape)
#             phiB = phi_B_numeric_world(P).reshape(X.shape)

#             plotter = pv.Plotter()
#             # φA=0
#             if (phiA.min() <= 0.0) and (phiA.max() >= 0.0):
#                 gridA = pv.ImageData()
#                 gridA.dimensions = np.array(phiA.shape)
#                 gridA.spacing   = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
#                 gridA.origin    = (x[0], y[0], z[0])
#                 gridA.point_data["phiA"] = phiA.flatten(order="F")
#                 contA = gridA.contour(isosurfaces=[0.0], scalars="phiA")
#                 plotter.add_mesh(contA, color="royalblue", opacity=0.55)
#             # φB=0
#             if (phiB.min() <= 0.0) and (phiB.max() >= 0.0):
#                 gridB = pv.ImageData()
#                 gridB.dimensions = np.array(phiB.shape)
#                 gridB.spacing   = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
#                 gridB.origin    = (x[0], y[0], z[0])
#                 gridB.point_data["phiB"] = phiB.flatten(order="F")
#                 contB = gridB.contour(isosurfaces=[0.0], scalars="phiB")
#                 plotter.add_mesh(contB, color="tomato", opacity=0.55)

#             # wireframe voxels (no pair lines)
#             MAX_DRAW = 1500
#             CdrawA = keptA[max_refine]; stepA = max(1, len(CdrawA)//MAX_DRAW if len(CdrawA) else 1)
#             hxA, hyA, hzA = halfA[max_refine]
#             for i in range(0, len(CdrawA), stepA):
#                 cx, cy, cz = CdrawA[i]
#                 cube = pv.Cube(center=(cx, cy, cz),
#                                x_length=2*hxA, y_length=2*hyA, z_length=2*hzA)
#                 plotter.add_mesh(cube, style="wireframe", color="yellow", line_width=1.0, opacity=0.85)

#             CdrawB = keptB[max_refine]; stepB = max(1, len(CdrawB)//MAX_DRAW if len(CdrawB) else 1)
#             hxB, hyB, hzB = halfB[max_refine]
#             for i in range(0, len(CdrawB), stepB):
#                 cx, cy, cz = CdrawB[i]
#                 cube = pv.Cube(center=(cx, cy, cz),
#                                x_length=2*hxB, y_length=2*hyB, z_length=2*hzB)
#                 plotter.add_mesh(cube, style="wireframe", color="cyan", line_width=1.0, opacity=0.85)

#             plotter.show_bounds(grid='front', location='outer', color='black',
#                                 xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
#             plotter.add_axes(); plotter.add_title("Two-Sided Broad-Phase: A(yellow) / B(cyan)")
#             plotter.camera_position = 'iso'
#             plotter.show()
#         except Exception as e:
#             print("[octree viz] skipped:", e)

#         # level diagnostics
#         try:
#             import matplotlib.pyplot as plt
#             levels = sorted(keptA.keys())
#             cntA = [len(keptA[L]) for L in levels]
#             cntB = [len(keptB[L]) for L in levels]
#             lowsA  = [intervalsA[L][0] for L in levels]
#             uppsA  = [intervalsA[L][1] for L in levels]
#             midsA  = [0.5*(l+u) for l,u in zip(lowsA, uppsA)]
#             lowsB  = [intervalsB[L][0] for L in levels]
#             uppsB  = [intervalsB[L][1] for L in levels]
#             midsB  = [0.5*(l+u) for l,u in zip(lowsB, uppsB)]
#             fig, axs = plt.subplots(1,3, figsize=(14,4))
#             axs[0].plot(levels, cntA, marker='o', label='A kept')
#             axs[0].plot(levels, cntB, marker='s', label='B kept')
#             axs[0].set_xlabel("L"); axs[0].set_ylabel("# voxels kept"); axs[0].legend(); axs[0].grid(alpha=0.3)
#             axs[1].plot(levels, lowsA, marker='v', label='A lower')
#             axs[1].plot(levels, uppsA, marker='^', label='A upper')
#             axs[1].plot(levels, midsA, marker='o', linestyle='--', label='A mid')
#             axs[1].set_xlabel("L"); axs[1].set_ylabel("Bracket on min φ_B"); axs[1].legend(); axs[1].grid(alpha=0.3)
#             axs[2].plot(levels, lowsB, marker='v', label='B lower')
#             axs[2].plot(levels, uppsB, marker='^', label='B upper')
#             axs[2].plot(levels, midsB, marker='o', linestyle='--', label='B mid')
#             axs[2].set_xlabel("L"); axs[2].set_ylabel("Bracket on min φ_A"); axs[2].legend(); axs[2].grid(alpha=0.3)
#             plt.suptitle(f"MNN pairs @final: {len(mnn_pairs)} (top-1 shown: {len(C_A_top)})")
#             plt.show()
#         except Exception as e:
#             print("[octree plots] skipped:", e)

#     # =============== ONLINE: penalized KKT (seeded by top-1 pair) =========================
#     # CSDL SDFs in world (m)
#     def phi_A_world_m(p_w: csdl.Variable) -> csdl.Variable:
#         R_wb = quat_to_rm_csdl(quat_wxyz)
#         R_bw = csdl.transpose(R_wb)
#         if p_w.shape == (3,):
#             rel_w  = p_w - pos_w_m
#             x_b_cm = 100.0 * csdl.matvec(R_bw, rel_w)
#             return phi_A_body_cm(x_b_cm) / 100.0
#         K = p_w.shape[0]
#         pos_tiled = csdl.expand(pos_w_m, out_shape=(K,3), action='i->ji')
#         rel_w  = p_w - pos_tiled
#         x_b_cm = 100.0 * csdl.matmat(rel_w, csdl.transpose(R_bw))
#         return phi_A_body_cm(x_b_cm) / 100.0

#     # penalty KKT with independent seeds for a,b
#     def penalty_kkt_pair(phi_A, phi_B, a0_np, b0_np, gamma=50.0, newton_tol=1e-8, name="penalty_kkt"):
#         a = csdl.Variable(name=f"{name}:a", shape=(3,), value=a0_np)
#         b = csdl.Variable(name=f"{name}:b", shape=(3,), value=b0_np)
#         phiA = phi_A(a);  phiB = phi_B(b)
#         gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
#         gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))
#         R_a = (a - b) + gamma * phiA * gA
#         R_b = (b - a) + gamma * phiB * gB
#         solver = csdl.nonlinear_solvers.Newton(name, tolerance=newton_tol)
#         solver.add_state(a, R_a, initial_value=a0_np)
#         solver.add_state(b, R_b, initial_value=b0_np)
#         solver.run()
#         m = 0.5*(a+b)
#         F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.05
#         #diff = a - b
#         #gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)
#         return m, F_star, a, b

#     # choose seeds
#     if len(C_A_top) == 1:
#         a0 = C_A_top[0]; b0 = C_B_top[0]
#         # print("Phi_A_top:", phi_A_numeric_world(a0)); print("Phi_B_top:", phi_B_numeric_world(b0));
#         # print("Top-1 MNN pair:", a0, b0)
#         # print("Midpoint C:", 0.5*(a0+b0))
#         # --- pick gamma from seed balance ---
#         a0 = C_A_top[0]; b0 = C_B_top[0]
#         d0 = float(np.linalg.norm(a0 - b0))
#         phiA0 = phi_A_numeric_world(a0)
#         phiB0 = phi_B_numeric_world(b0)
#         phi_bar = max(1e-4, 0.5*(abs(phiA0) + abs(phiB0)))   # guard
#         gamma_bal = np.clip(d0 / phi_bar, 5.0, 300.0)        # seed-balanced gamma

#         m, F_star, a, b = penalty_kkt_pair(phi_A_world_m, phi_B_world_m,
#                                                     a0, b0,
#                                                     gamma=gamma_bal, newton_tol=newton_tol,
#                                                     name="collision_kkt_gbal")
#         # # --- (optional) two-stage continuation for robustness ---
#         # gammas = [0.5*gamma_bal, gamma_bal, 0.5*(gamma_bal+gamma), gamma]  # last is your target (e.g., 50)
#         # a_init, b_init = a0, b0
#         # for g in gammas:
#         #     m, F_star, a, b, pair_gap = penalty_kkt_pair(phi_A_world_m, phi_B_world_m,
#         #                                                 a_init, b_init,
#         #                                                 gamma=g, newton_tol=newton_tol,
#         #                                                 name=f"collision_kkt_g{g:.1f}")
#         #     # warm start next stage
#         #     a_init = np.asarray(a.value).reshape(3)
#         #     b_init = np.asarray(b.value).reshape(3)

#     else:
#         # fallback: midpoint between drone and gate center if no pairs
#         mid = 0.5*(p_w + b_seed)
#         print("Midpoint fallback:", mid)
#         m, F_star, a, b = penalty_kkt_pair(phi_A_world_m, phi_B_world_m, mid, mid,
#                                                      gamma=gamma, newton_tol=newton_tol, name="collision_kkt_fallback")

#     # optional KKT plotting
#     if plot_kkt:
#         try:
#             import pyvista as pv
#             # quick grids (drone tight; gate broader)
#             pos_np = np.asarray(pos_w_m.value)
#             d_half = 0.22
#             bounds_drone = ((pos_np[0]-d_half, pos_np[0]+d_half),
#                             (pos_np[1]-d_half, pos_np[1]+d_half),
#                             (pos_np[2]-d_half, pos_np[2]+d_half))
#             g_half = np.array([1.0, 1.5, 1.5])  # rough sphere/box max with margin
#             gcen  = b_seed
#             bounds_gate = ((gcen[0]-g_half[0], gcen[0]+g_half[0]),
#                            (gcen[1]-g_half[1], gcen[1]+g_half[1]),
#                            (gcen[2]-g_half[2], gcen[2]+g_half[2]))
#             def contour_zero(phi_fn, bounds, dx, scal_name):
#                 if not hasattr(pv, "ImageData"): raise RuntimeError("PyVista ImageData missing")
#                 (xmin,xmax),(ymin,ymax),(zmin,zmax) = bounds
#                 nx = int(np.floor((xmax-xmin)/dx))+1
#                 ny = int(np.floor((ymax-ymin)/dx))+1
#                 nz = int(np.floor((zmax-zmin)/dx))+1
#                 grid = pv.ImageData()
#                 grid.dimensions = (nx,ny,nz); grid.origin=(xmin,ymin,zmin); grid.spacing=(dx,dx,dx)
#                 P = np.asarray(grid.points)
#                 vals = phi_fn(csdl.Variable(value=P.astype(float))).value.reshape(-1)
#                 grid[scal_name] = vals
#                 return grid.contour(isosurfaces=[0.0], scalars=scal_name)

#             surf_A = contour_zero(phi_A_world_m, bounds_drone, 0.01, "phiA")
#             surf_B = contour_zero(phi_B_world_m, bounds_gate,  0.04, "phiB")

#             a_np = np.asarray(a.value).reshape(3)
#             b_np = np.asarray(b.value).reshape(3)
#             m_np = np.asarray(m.value).reshape(3)
#             s_a = pv.Sphere(radius=0.010, center=a_np)
#             s_b = pv.Sphere(radius=0.010, center=b_np)
#             s_m = pv.Sphere(radius=0.008, center=m_np)

#             p = pv.Plotter()
#             p.add_mesh(surf_A, color="#1f77b4", opacity=0.75, smooth_shading=True, label="Drone (A=0)")
#             p.add_mesh(surf_B, color="#f46c6c", opacity=0.35, smooth_shading=True, label="Gate (B=0)")
#             p.add_mesh(s_a, color="#bc2525"); p.add_mesh(s_b, color="#9467bd"); p.add_mesh(s_m, color="#ffdd57")
#             p.add_axes(line_width=2); p.add_legend(bcolor="black", border=True)
#             p.set_background("black", top="dimgray"); p.camera_position="iso"
#             p.show(title="Penalty KKT: φA=0, φB=0, witnesses a/b, midpoint m")
#         except Exception as e:
#             print("[kkt viz] skipped:", e)

#     # return dict(
#     #     keptA=keptA, halfA=halfA, intervalsA=intervalsA,
#     #     keptB=keptB, halfB=halfB, intervalsB=intervalsB,
#     #     mnn_pairs=mnn_pairs, C_A_top=C_A_top, C_B_top=C_B_top,
#     #     m=m, F_star=F_star, a=a, b=b, pair_gap=pair_gap
#     # )
#     return m, F_star, a, b


# =============== END HYBRID METHOD ===============




#SDF = Callable[[csdl.Variable], csdl.Variable]

#_DEF_EPS = 1e-12

# # Collision check similar to paper
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     rho: float = 20.0,             # higher = sharper max; start 5–20
#     return_all: bool = False,
#     newton_name: str = "collision_check_smooth",
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Differentiable SDF-SDF proximity/collision check

#     Steps:
#       1) Define smooth joint field F(x) = max(phi_A(x), phi_B(x)) 
#       2) Do K fixed 'pull' steps: x <- x - eta * ∇F / ||∇F||.
#       3) Project once from x_K to each surface:
#            a = x_K - phi_A(x_K) / ||∇phi_A(x_K)||^2 * ∇phi_A(x_K)
#            b = x_K - phi_B(x_K) / ||∇phi_B(x_K)||^2 * ∇phi_B(x_K)
#       4) Outputs: gap = ||a - b||  (always >= 0), F_star = F(x_K),
#          and optionally x_K, a, b.

#     Returns:
#       If return_all: (x_K, F_star, a, b, gap), else just F.
#     """
#     # ---- seed x0 as leaf variable ----
#     if isinstance(x0, csdl.Variable):
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     else:
#         x0_np = np.asarray(x0, float).reshape(3,)
#     #x = csdl.Variable(name=f"{newton_name}:x0", shape=(3,), value=x0_np)
#     x = x0
#     # ---- finite gradient steps on F(x) = max(phi_A, phi_B) ----

#     #for c_i in (0.9, 0.5, 0.25): 
#     #c_i = 1.0

#       # inside each pull iteration
#     FA = phi_A(x)
#     FB = phi_B(x)
#     F  = csdl.maximum(FA, FB, rho=rho)

#     gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))
#     gF_norm = csdl.norm(gF) + _DEF_EPS

#     # adaptive step length: eta = min(c * |F|, eta_max)
#     #eta = csdl.minimum(c_i * csdl.absolute(F), eta_max)   # choose c_i per pull, e.g. 0.7, 0.35, 0.2

#     x = x - (gF / gF_norm)

#     xK = x
#     F_star = csdl.maximum(phi_A(xK), phi_B(xK), rho=rho) -1.0

    
#     #---- two-step projection to each surface from xK ----
#     if return_all:
#         # ---------- helpers: safe L2 norm and unit ----------
#         def safe_norm2(v, eps=_DEF_EPS):
#             ax = csdl.absolute(v)
#             vmax01 = csdl.maximum(ax[0], ax[1])
#             vmax   = csdl.maximum(vmax01, ax[2])            # max(|v_i|)
#             scale  = csdl.maximum(vmax, csdl.Variable(value = 1.0))                # keep >= 1 to avoid div-by-zero
#             v_sc   = v / scale
#             return scale * csdl.sqrt(csdl.vdot(v_sc, v_sc) + eps)

#         def safe_unit(v, eps=_DEF_EPS):
#             return v / (safe_norm2(v, eps) + eps)

#         # ---- keep a copy of the candidate x* for residuals ----
#         x_cand = xK
#         np.set_printoptions(precision=6, suppress=True)

#         def _inf_norm(u):      # for numpy arrays (after .value)
#             return float(np.max(np.abs(u)))


#         x_cand = xK

#         # ===== A: first projection step =====
#         print("[PIN] before phi_A(x)")
#         FA0 = phi_A(x_cand)
#         print("[PIN] after  phi_A(x) | FA(x)=", float(FA0.value))

#         print("[PIN] before gradA(x)")
#         gA0 = csdl.reshape(csdl.derivative(ofs=FA0, wrts=x_cand), (3,))
#         print("[PIN] after  gradA(x) | max|gA|=", float(np.max(np.abs(gA0.value))))

#         gA0n = safe_norm2(gA0) + _DEF_EPS
#         sA   = csdl.absolute(FA0) / gA0n
#         a1   = x_cand - (FA0 * gA0) / (gA0n * gA0n)

#         print("[DBG] A0  | FA(x)       =", float(FA0.value),
#                 "||gA||    =", float(gA0n.value),
#                 "max|gA|   =", float(np.max(np.abs(gA0.value))),
#                 "step      =", float(sA.value),
#                 "||x||inf  =", _inf_norm(x_cand.value),
#                 "||a1||inf =", _inf_norm(a1.value))

#         # This call is a common overflow point if a1 jumped far:
#         FA1 = phi_A(a1)
#         print("[DBG] A1  | FA(a1)      =", float(FA1.value))

#         # ===== A: second projection step =====
#         gA1  = csdl.reshape(csdl.derivative(ofs=FA1, wrts=a1), (3,))
#         gA1n = safe_norm2(gA1) + _DEF_EPS
#         a    = a1 - (FA1 * gA1) / (gA1n * gA1n)

#         print("[DBG] A2  | ||gA(a1)||  =", float(gA1n.value),
#                 "max|gA(a1)|=", float(np.max(np.abs(gA1.value))),
#                 "||a||inf   =", _inf_norm(a.value))

#         # ===== B: first projection step =====
#         FB0  = phi_B(x_cand)
#         gB0  = csdl.reshape(csdl.derivative(ofs=FB0, wrts=x_cand), (3,))
#         gB0n = safe_norm2(gB0) + _DEF_EPS
#         sB   = csdl.absolute(FB0) / gB0n
#         b1   = x_cand - (FB0 * gB0) / (gB0n * gB0n)

#         print("[DBG] B0  | FB(x)       =", float(FB0.value),
#                 "||gB||    =", float(gB0n.value),
#                 "max|gB|   =", float(np.max(np.abs(gB0.value))),
#                 "step      =", float(sB.value),
#                 "||x||inf  =", _inf_norm(x_cand.value),
#                 "||b1||inf =", _inf_norm(b1.value))

#         # Another common overflow point:
#         FB1 = phi_B(b1)
#         print("[DBG] B1  | FB(b1)      =", float(FB1.value))

#         # ===== B: second projection step =====
#         gB1  = csdl.reshape(csdl.derivative(ofs=FB1, wrts=b1), (3,))
#         gB1n = safe_norm2(gB1) + _DEF_EPS
#         b    = b1 - (FB1 * gB1) / (gB1n * gB1n)

#         print("[DBG] B2  | ||gB(b1)||  =", float(gB1n.value),
#                 "max|gB(b1)|=", float(np.max(np.abs(gB1.value))),
#                 "||b||inf   =", _inf_norm(b.value))

#         # ---- gap (optional) ----
#         diff = a - b
#         gap  = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)
#         print("[DBG] GAP | ||a-b||     =", float(gap.value))

#         # ===== residuals =====
#         FA_x = phi_A(x_cand); FB_x = phi_B(x_cand)
#         r_eq = csdl.absolute(FA_x - FB_x)
#         print("[DBG] EQ  | |FA-FB|     =", float(r_eq.value))

#         FA_a = phi_A(a); gA_a = csdl.reshape(csdl.derivative(ofs=FA_a, wrts=a), (3,))
#         FB_b = phi_B(b); gB_b = csdl.reshape(csdl.derivative(ofs=FB_b, wrts=b), (3,))
#         nA   = gA_a / (safe_norm2(gA_a) + _DEF_EPS)
#         nB   = gB_b / (safe_norm2(gB_b) + _DEF_EPS)

#         print("[DBG] NRM | ||gA(a)||   =", float(safe_norm2(gA_a).value),
#                 "||gB(b)|| =", float(safe_norm2(gB_b).value),
#                 "max|gA(a)|=", float(np.max(np.abs(gA_a.value))),
#                 "max|gB(b)|=", float(np.max(np.abs(gB_b.value))))

#         r_dir = 0.5 * safe_norm2(nA + nB)
#         print("[DBG] DIR | r_dir       =", float(r_dir.value))

#         r_eik_A = csdl.absolute(safe_norm2(gA_a) - 1.0)
#         r_eik_B = csdl.absolute(safe_norm2(gB_b) - 1.0)
#         print("[DBG] EIK | A=", float(r_eik_A.value), "B=", float(r_eik_B.value))


#     if return_all:
#         return xK, F_star, a, b
#     return xK, F_star





# # It works?? Substitute lagrange multipliers
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_stationarity",
#     return_all: bool = False,
#     normalize_grads: bool = True,
#     grad_floor: float = 1e-8,        # guard for ||∇φ||
#     reg_pos: float = 1e-6,           # tiny Tikhonov regularization (anchors at seeds)
#     surf_weight: float = 1.0,        # penalty strength pushing a,b to their surfaces
#     beta_scale_with_grad: bool = True # scale penalty by ||∇φ|| if not normalizing
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:

#     # ----- numeric seeds -----
#     try:
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     except AttributeError:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np = x0_np.copy()
#     b0_np = x0_np.copy()

#     # constants / guards
#     _eps = 1e-12

#     # keep copies of the seeds as constants for regularization anchors
#     a0_const = csdl.Variable(value=a0_np)
#     b0_const = csdl.Variable(value=b0_np)

#     ones3 = csdl.Variable(value=np.ones(3))

#     # ----- states (only a and b) -----
#     a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
#     b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

#     # ----- SDF values & (safe) normals -----
#     # A
#     phiA_a = phi_A(a)                                       # scalar
#     gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     if normalize_grads:
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#         betaA   = surf_weight
#     else:
#         # still guard the zero-gradient case
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#         betaA   = surf_weight * (gA_norm if beta_scale_with_grad else 1.0)

#     # B
#     phiB_b = phi_B(b)                                       # scalar
#     gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))
#     if normalize_grads:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#         betaB   = surf_weight
#     else:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#         betaB   = surf_weight * (gB_norm if beta_scale_with_grad else 1.0)

#     # Reshape scalars for vector scaling
#     phiA_as3 = csdl.reshape(phiA_a, (1,)) * ones3
#     phiB_as3 = csdl.reshape(phiB_b, (1,)) * ones3

#     # ----- residuals (penalized stationarity, no λ/μ states) -----
#     # Intuition: original KKT had (a - b) + λ nA = 0, (a - b) - μ nB = 0 with φ_A(a)=0, φ_B(b)=0.
#     # Here we replace λ ≈ -β φ_A(a), μ ≈  β φ_B(b), which pushes a,b to their surfaces and couples them.
#     RS_A = (a - b) + betaA * (phiA_as3 * nA) + reg_pos * (a - a0_const)  # (3,)
#     RS_B = (a - b) - betaB * (phiB_as3 * nB) + reg_pos * (b - b0_const)  # (3,)

#     # ----- Newton solve -----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     # (Optional) If your CSDL build supports it, you can also set a safe max iters, damping, etc.
#     # e.g., solver.options['max_iter'] = 80
#     solver.add_state(a, RS_A, initial_value=a0_np)   # 3 eqs
#     solver.add_state(b, RS_B, initial_value=b0_np)   # 3 eqs
#     solver.run()

#     # ---------- helpers (shape-safe) ----------
#     def _snap_once(p, phi, grad_floor=1e-8):
#         # One Newton projection step: p <- p - φ * n, with unit-normal and floor
#         val = phi(p)                                              # scalar
#         g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
#         n   = g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)
#         return p - csdl.reshape(val, (1,)) * n                    # (3,)

#     def _snap_k(p0, phi, k=2, grad_floor=1e-8):
#         p = p0
#         for _ in range(k):
#             p = _snap_once(p, phi, grad_floor)
#         return p

#     def _unit(v, eps=1e-12):
#         return v / (csdl.sqrt(csdl.vdot(v, v)) + eps)

#     def _normal_at(p, phi, grad_floor=1e-8):
#         val = phi(p)  # scalar (unused; ensures correct graph deps)
#         g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p), (3,))
#         return g / csdl.sqrt(csdl.vdot(g, g) + grad_floor)

#     # ---------- alternating-projection refinement ----------
#     # Idea: land each point on the hemisphere facing the *other* body by projecting
#     # from the other body’s point. Do both orders and keep the better one.

#     # Option A: start from b toward A, then from a toward B (B→A→B)
#     a_A1 = _snap_k(b, phi_A, k=2, grad_floor=grad_floor)         # project B's point onto A
#     b_A1 = _snap_k(a_A1, phi_B, k=2, grad_floor=grad_floor)      # then project that onto B

#     # Option B: start from a toward B, then from b toward A (A→B→A)
#     b_B1 = _snap_k(a, phi_B, k=2, grad_floor=grad_floor)
#     a_B1 = _snap_k(b_B1, phi_A, k=2, grad_floor=grad_floor)

#     # Score both candidates by (i) smaller gap and (ii) better normal alignment
#     def _score_pair(aP, bP, phiA, phiB):
#         t   = _unit(aP - bP)
#         nA  = _normal_at(aP, phiA, grad_floor)
#         nB  = _normal_at(bP, phiB, grad_floor)
#         gap = csdl.sqrt(csdl.vdot(aP - bP, aP - bP) + 1e-12)
#         # alignment: both outward normals should align with chord (>=0, closer to 1 is better)
#         align = 0.5 * (csdl.vdot(nA, t) + csdl.vdot(nB, t))       # scalar
#         return gap, align

#     gap_A, align_A = _score_pair(a_A1, b_A1, phi_A, phi_B)
#     gap_B, align_B = _score_pair(a_B1, b_B1, phi_A, phi_B)

#     # Choose the better pair: prefer smaller gap; break ties by larger alignment
#     # (CSDL has no conditionals; pick both and *use the one you return* in Python flow)
#     use_A = True  # Python-level choice is fine since you're building the graph once per call
#     # You can fetch initial seeds’ numpy to decide, but simplest is:
#     #   if you want fully CSDL, keep both and return both to caller to choose.
#     try:
#         # If shapes were numeric Variables, take a quick numpy read for the decision:
#         use_A = (float(gap_A.value) < float(gap_B.value)) or \
#                 (abs(float(gap_A.value) - float(gap_B.value)) < 1e-9 and float(align_A.value) >= float(align_B.value))
#     except Exception:
#         # Fallback: prefer A by default
#         use_A = True

#     a_ref = a_A1 if use_A else a_B1
#     b_ref = b_A1 if use_A else b_B1

#     # # Optional tiny clean-up snap to ensure |φ| ~ 0 after the alternations
#     # a_ref = _snap_once(a_ref, phi_A, grad_floor)
#     # b_ref = _snap_once(b_ref, phi_B, grad_floor)

#     # ---------- replace outputs ----------
#     a = a_ref
#     b = b_ref
#     diff = a - b
#     gap  = csdl.sqrt(csdl.vdot(diff, diff) + 1e-12)
#     x_mid  = 0.5 * (a + b)
#     F_star = csdl.maximum(phi_A(x_mid), phi_B(x_mid), rho=20.0)- 0.07

#     if return_all:
#         return x_mid, F_star, a, b, gap
#     return gap



#IDK ANYMORE KKT AGAIN
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_stationarity",
#     return_all: bool = False,
#     normalize_grads: bool = True,
#     grad_floor: float = 1e-8,        # guard for ||∇φ||
#     reg_pos: float = 1e-6            # tiny Tikhonov regularization in stationarity eqs
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:

#     # ----- numeric seeds -----
#     try:
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     except AttributeError:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np   = x0_np.copy()
#     b0_np   = x0_np.copy()
#     lam0_np = np.array([0.0], dtype=float)
#     mu0_np  = np.array([0.0], dtype=float)

#     # constants / guards
#     _eps = 1e-12 if "_DEF_EPS" in globals() else 1e-12

#     # keep copies of the seeds as constants for regularization anchors
#     a0_const = csdl.Variable(value=a0_np)
#     b0_const = csdl.Variable(value=b0_np)

#     # ----- states -----
#     a   = csdl.Variable(name=f"{newton_name}:a",   shape=(3,), value=a0_np)
#     b   = csdl.Variable(name=f"{newton_name}:b",   shape=(3,), value=b0_np)
#     lam = csdl.Variable(name=f"{newton_name}:lam", shape=(1,), value=lam0_np)
#     mu  = csdl.Variable(name=f"{newton_name}:mu",  shape=(1,), value=mu0_np)

#     # ----- SDF values & (safe) normals -----
#     # A
#     phiA_a = phi_A(a)                                        # scalar
#     gA     = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     if normalize_grads:
#         gA_norm = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / gA_norm
#     else:
#         # still guard the zero-gradient case
#         scale   = csdl.sqrt(csdl.vdot(gA, gA) + grad_floor)
#         nA      = gA / scale

#     # B
#     phiB_b = phi_B(b)                                        # scalar
#     gB     = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))
#     if normalize_grads:
#         gB_norm = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / gB_norm
#     else:
#         scale   = csdl.sqrt(csdl.vdot(gB, gB) + grad_floor)
#         nB      = gB / scale

#     # broadcast λ, μ to 3-vectors
#     ones3 = csdl.Variable(value=np.ones(3))
#     lam3  = csdl.reshape(csdl.tensordot(lam, ones3, axes=None), (3,))
#     mu3   = csdl.reshape(csdl.tensordot(mu,  ones3, axes=None), (3,))

#     # ----- residuals -----
#     # Surface constraints
#     RA = csdl.reshape(phiA_a, (1,))                    # φ_A(a) = 0
#     RB = csdl.reshape(phiB_b, (1,))                    # φ_B(b) = 0

#     # Stationarity with tiny Tikhonov (anchors at seeds)
#     #   (a - b) + λ nA + reg*(a - a0) = 0
#     #   (a - b) - μ nB + reg*(b - b0) = 0
#     RS_A = (a - b) + lam3 * nA + reg_pos * (a - a0_const)   # (3,)
#     RS_B = (a - b) - mu3  * nB + reg_pos * (b - b0_const)   # (3,)

#     # ----- Newton solve -----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a,   RS_A, initial_value=a0_np)   # 3 eqs
#     solver.add_state(b,   RS_B, initial_value=b0_np)   # 3 eqs
#     solver.add_state(lam, RA,   initial_value=lam0_np) # 1 eq
#     solver.add_state(mu,  RB,   initial_value=mu0_np)  # 1 eq
#     solver.run()

#     # ----- outputs -----
#     diff = a - b
#     gap  = csdl.sqrt(csdl.vdot(diff, diff) + _eps)

#     # mid-point & smooth max (kept for API compatibility)
#     x_mid  = 0.5 * (a + b)
#     F_star = csdl.maximum(phi_A(x_mid), phi_B(x_mid), rho=20.0)

#     if return_all:
#         return x_mid, F_star, a, b, gap
#     return gap











# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     gamma: float = 50.0,                 # penalty strength (try 1–20 depending on units)
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_penalty",
#     return_all: bool = False,
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Fully differentiable collision check via the penalized closest-pair stationarity:
#         (a - b) + gamma * phi_A(a) * gA_hat = 0
#         (b - a) + gamma * phi_B(b) * gB_hat = 0
#     Returns pair_gap by default; with return_all=True returns (m, F_star, a, b, pair_gap).
#     """

#     # ---- numeric seeds (leaf states) ----
#     m0 = x0.value.reshape(3,) if isinstance(x0, csdl.Variable) else np.asarray(x0, float).reshape(3,)
#     a0_np = m0.copy()
#     b0_np = m0.copy()

#     a = csdl.Variable(name=f"{newton_name}:a", shape=(3,), value=a0_np)
#     b = csdl.Variable(name=f"{newton_name}:b", shape=(3,), value=b0_np)

#     # φ and gradients
#     phiA = phi_A(a)  # scalar
#     phiB = phi_B(b)  # scalar

#     gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))

#     # Stationarity residuals (6 equations,3 each)
#     R_a = (a - b) + gamma * phiA * gA   # (3,)
#     R_b = (b - a) + gamma * phiB * gB   # (3,)

#     # Newton solve 
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a, R_a, initial_value=a0_np)
#     solver.add_state(b, R_b, initial_value=b0_np)
#     solver.run()

#     m = 0.5 * (a + b)

#     F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.0507

#     if return_all:
#         # Additional Outputs
#         diff = a - b
#         pair_gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#         return  m, F_star, a, b, pair_gap
#     return F_star


# =================== KKT ATTEMPT ====================
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     gamma: float = 50.0,                 # penalty strength (try 1–20 depending on units)
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_penalty",
#     return_all: bool = False,
# ) -> Tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Fully differentiable collision check via the penalized closest-pair stationarity:
#         (a - b) + gamma * phi_A(a) * gA_hat = 0
#         (b - a) + gamma * phi_B(b) * gB_hat = 0
#     Returns pair_gap by default; with return_all=True returns (m, F_star, a, b, pair_gap).
#     """

#     # ---- numeric seeds (leaf states) ----
#     m0 = x0.value.reshape(3,) if isinstance(x0, csdl.Variable) else np.asarray(x0, float).reshape(3,)
#     a0_np = m0.copy()
#     b0_np = m0.copy()

#     # Unknowns
#     a = csdl.Variable((3,), value=a0_np)
#     b = csdl.Variable((3,), value=b0_np)
#     lamA = csdl.Variable(value=0.0)
#     lamB = csdl.Variable(value=0.0)

#     phiA = phi_A(a); gA = csdl.reshape(csdl.derivative(ofs=phiA, wrts=a), (3,))
#     phiB = phi_B(b); gB = csdl.reshape(csdl.derivative(ofs=phiB, wrts=b), (3,))

#     R_a = (a - b) + lamA * gA
#     R_b = (b - a) + lamB * gB
#     C_A = phiA               # equality constraint
#     C_B = phiB

#     solver = csdl.nonlinear_solvers.Newton("closest_pair_kkt", tolerance=1e-8)
#     solver.add_state(a, R_a)
#     solver.add_state(b, R_b)
#     solver.add_state(lamA, C_A)  # enforces phi_A(a)=0
#     solver.add_state(lamB, C_B)  # enforces phi_B(b)=0
#     solver.run()


#     m = 0.5 * (a + b)

#     F_star = csdl.maximum(phi_A(m), phi_B(m), rho=20.0) - 0.0507

#     if return_all:
#         # Additional Outputs
#         diff = a - b
#         pair_gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#         return  m, F_star, a, b, pair_gap
#     return F_star

# =================== END KKT ATTEMPT ====================






















# Leftover helper for below
# def _project_from_x(x: csdl.Variable, phi_x: csdl.Variable, grad_x: csdl.Variable,
#                     eps: float = _DEF_EPS) -> csdl.Variable:
#     """First-order projection of x onto the zero level set of an SDF.
#     a = x - phi(x) * grad/||grad||^2
#     """
#     denom = csdl.sum(grad_x * grad_x) + eps
#     return x - (phi_x * grad_x) / denom


# Gradient of F formula (issues with deep collisions)
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check",
#     return_all: bool = False,
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Perform collision check between two SDFs.

#     Parameters
#     ----------
#     phi_A, phi_B : callable
#         Functions mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
#     x0 : np.ndarray | csdl.Variable, shape (3,)
#         Initial seed for the Newton solve. May be a NumPy array or a CSDL Variable.
#     newton_tol : float
#         Convergence tolerance.
#     newton_name : str
#         Name for the CSDL Newton solver instance.
#     return_all : bool
#         If True, return tuple (x_star, F_star, a, b, pair_gap).
#         Else return just pair_gap.
#     """
#     if isinstance(x0, csdl.Variable):
#         x = x0
#         use_initial_value = False
#         init_val = None
#     else:
#         x = csdl.Variable(name=f"{newton_name}:x", shape=(3,), value=np.asarray(x0, dtype=float))
#         use_initial_value = True
#         init_val = np.asarray(x0, dtype=float)

#     phiA_x = phi_A(x)
#     phiB_x = phi_B(x)
#     F = csdl.maximum(phiA_x, phiB_x)

#     gF = csdl.reshape(csdl.derivative(ofs=F, wrts=x), (3,))

#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     if use_initial_value:
#         solver.add_state(x, gF, initial_value=init_val)
#     else:
#         solver.add_state(x, gF)
#     solver.run()

#     phiA_x = phi_A(x)
#     phiB_x = phi_B(x)
#     gA = csdl.reshape(csdl.derivative(ofs=phiA_x, wrts=x), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB_x, wrts=x), (3,))

#     a = _project_from_x(x, phiA_x, gA)
#     b = _project_from_x(x, phiB_x, gB)
#     pair_gap = csdl.norm(a - b)

#     if return_all:
#         return x, F, a, b, pair_gap
#     return pair_gap

# __all__ = ["collision_check"]


# KKT approach (issues with Nan/Inf/Singular Matrix)
# def collision_check(
#     phi_A: SDF,
#     phi_B: SDF,
#     x0: Union[np.ndarray, csdl.Variable],
#     *,
#     newton_tol: float = 1e-8,
#     newton_name: str = "collision_check_kkt",
#     return_all: bool = False,
#     normalize_grads: bool = False,
# ) -> tuple[csdl.Variable, ...] | csdl.Variable:
#     """
#     Closest-pair collision check via KKT system:

#         a - x - λ_A ∇φ_A(a) = 0
#         b - x - λ_B ∇φ_B(b) = 0
#         φ_A(a) = 0
#         φ_B(b) = 0
#         x - (a + b)/2 = 0

#     Parameters
#     ----------
#     phi_A, phi_B : callable
#         SDFs mapping csdl.Variable(shape=(3,)) -> scalar csdl.Variable.
#     x0 : np.ndarray | csdl.Variable
#         Numeric seed for the midpoint state x (if csdl.Variable, its .value is used).
#     newton_tol : float
#         Newton tolerance.
#     newton_name : str
#         Name for the Newton solver instance.
#     return_all : bool
#         If True, returns (x_star, F_star, a_star, b_star, pair_gap).
#         Else returns just pair_gap.
#     normalize_grads : bool
#         If True, uses unit normals g/||g|| in the KKT equations.

#     Returns
#     -------
#     tuple or csdl.Variable
#         If return_all=True: (x, F_star, a, b, pair_gap).
#         Else: pair_gap.
#     """
#     # ---- numeric seeds (keep states as leaf variables) ----
#     if isinstance(x0, csdl.Variable):
#         x0_np = np.asarray(x0.value, float).reshape(3,)
#     else:
#         x0_np = np.asarray(x0, float).reshape(3,)

#     a0_np   = x0_np.copy()
#     b0_np   = x0_np.copy()
#     lamA0_np = np.array([0.0], dtype=float)
#     lamB0_np = np.array([0.0], dtype=float)

#     # ---- states ----
#     a    = csdl.Variable(name=f"{newton_name}:a",    shape=(3,), value=a0_np)
#     lamA = csdl.Variable(name=f"{newton_name}:lamA", shape=(1,), value=lamA0_np)
#     b    = csdl.Variable(name=f"{newton_name}:b",    shape=(3,), value=b0_np)
#     lamB = csdl.Variable(name=f"{newton_name}:lamB", shape=(1,), value=lamB0_np)
#     x    = csdl.Variable(name=f"{newton_name}:x",    shape=(3,), value=x0_np)

#     # ---- values & gradients on each surface point ----
#     phiA_a = phi_A(a)                              # scalar
#     phiB_b = phi_B(b)                              # scalar
#     gA = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a), (3,))
#     gB = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b), (3,))

#     if normalize_grads:
#         gA = gA / csdl.sqrt(csdl.vdot(gA, gA) + _DEF_EPS)
#         gB = gB / csdl.sqrt(csdl.vdot(gB, gB) + _DEF_EPS)

#     # broadcast λ to 3-vector
#     ones3 = csdl.Variable(value=np.ones(3))
#     lamA3 = csdl.reshape(csdl.tensordot(lamA, ones3, axes=None), (3,))
#     lamB3 = csdl.reshape(csdl.tensordot(lamB, ones3, axes=None), (3,))

#     # ---- residuals ----
#     RA1 = a - x - lamA3 * gA              # (3,)
#     RA2 = csdl.reshape(phiA_a, (1,))      # (1,)
#     RB1 = b - x - lamB3 * gB              # (3,)
#     RB2 = csdl.reshape(phiB_b, (1,))      # (1,)
#     RX  = x - 0.5*(a + b)                 # (3,)

#     # ---- Newton solve ----
#     solver = csdl.nonlinear_solvers.Newton(newton_name, tolerance=newton_tol)
#     solver.add_state(a,    RA1, initial_value=a0_np)
#     solver.add_state(lamA, RA2, initial_value=lamA0_np)
#     solver.add_state(b,    RB1, initial_value=b0_np)
#     solver.add_state(lamB, RB2, initial_value=lamB0_np)
#     solver.add_state(x,    RX,  initial_value=x0_np)
#     solver.run()

#     # ---- outputs ----
#     # pair gap (guarded)
#     diff = a - b
#     gap = csdl.sqrt(csdl.vdot(diff, diff) + _DEF_EPS)

#     # F_star for compatibility with your previous return tuple
#     F_star = csdl.maximum(phi_A(x), phi_B(x), rho=20.0)

#     if return_all:
#         return x, F_star, a, b, gap
#     return gap


# __all__ = [
#     "collision_check_kkt",
# ]

# # ------------------------------------------------------------
# # Two-phase collision solve with feasibility weighting
# # Phase 1: LM projection to each surface (gets |phi| ~ 0)
# # Phase 2: KKT LM with weighted feasibility (robust from far seeds)
# # ------------------------------------------------------------

# def collision_check_kkt_LM(
#     phi_A, phi_B, x0,
#     *,
#     # Phase-1 (projection) LM settings
#     proj_max_iter: int = 40,
#     proj_tol: float = 1e-12,
#     # Phase-2 (KKT) LM settings
#     kkt_max_iter: int = 80,
#     kkt_tol: float = 1e-12,
#     lm_mu0: float = 1e-2,
#     lm_mu_inc: float = 10.0,
#     lm_mu_dec: float = 1/3,
#     lm_eta: float = 1e-3,             # gain ratio acceptance threshold
#     # Feasibility weighting schedule (for Phase-2)
#     feas_weight_init: float = 50.0,    # start heavy so φ-terms dominate when far
#     feas_weight_decay: float = 0.5,    # multiply weight by this each schedule step
#     feas_taper_threshold: float = 1e-3,# once max(|φ|) below this, set weight=1
#     feas_schedule_steps: int = 3,      # how many weighted passes before w=1
#     # Misc
#     return_all: bool = False
# ):
#     """
#     Returns: gap (default)
#              or (x_mid, a, b, gap) if return_all=True
#     """

#     # ---------- utils to extract numpy ----------
#     def _to_np(x, shape):
#         try:
#             return np.asarray(x.value, float).reshape(*shape)
#         except Exception:
#             return np.asarray(x, float).reshape(*shape)

#     # ---------- Phase-1: LM projection to a surface ----------
#     def _lm_project(phi, p0_np):
#         mu = float(lm_mu0)
#         p = p0_np.copy()

#         def FJ(p_np):
#             p_v = csdl.Variable(value=p_np.reshape(3,))
#             val = phi(p_v)                                        # scalar
#             g   = csdl.reshape(csdl.derivative(ofs=val, wrts=p_v),(3,))
#             F   = np.array([float(val.value)], float)             # (1,)
#             J   = np.zeros((1,3), float); J[0,:] = _to_np(g,(3,))
#             return F, J

#         F, J = FJ(p); f2 = float(F @ F)
#         for _ in range(proj_max_iter):
#             JTJ = J.T @ J
#             g = J.T @ F
#             try:
#                 dp = -np.linalg.solve(JTJ + mu*np.eye(3), g).reshape(3,)
#             except np.linalg.LinAlgError:
#                 mu *= lm_mu_inc; continue
#             p_trial = p + dp
#             F_trial, J_trial = FJ(p_trial); f2_trial = float(F_trial @ F_trial)

#             # predicted reduction (quadratic model on 1D residual)
#             pred_red = - (dp @ g).item() - 0.5 * (dp @ (JTJ @ dp)).item()
#             rho = (f2 - f2_trial) / max(pred_red, 1e-16)

#             if rho > lm_eta and f2_trial < f2:
#                 p, F, J, f2 = p_trial, F_trial, J_trial, f2_trial
#                 mu = max(mu * lm_mu_dec, 1e-15)
#                 if f2 < proj_tol: break
#             else:
#                 mu *= lm_mu_inc
#         return p

#     # ---------- Phase-2: weighted KKT LM (one run) ----------
#     # Unknowns: a(3), b(3), lamA(1), lamB(1) -> x in R^8
#     def _kkt_LM(a_seed_np, b_seed_np, w_phi: float, max_iter: int, tol: float):
#         def pack(a, b, lamA, lamB):
#             x = np.zeros(8, float)
#             x[0:3] = a; x[3:6] = b; x[6] = lamA[0]; x[7] = lamB[0]
#             return x
#         def unpack(x):
#             return x[0:3], x[3:6], np.array([x[6]]), np.array([x[7]])

#         mu = float(lm_mu0)
#         x = pack(a_seed_np, b_seed_np, np.array([0.0]), np.array([0.0]))

#         def FJ(x_vec):
#             a_np, b_np, lamA_np, lamB_np = unpack(x_vec)

#             a_v = csdl.Variable(value=a_np.reshape(3,))
#             b_v = csdl.Variable(value=b_np.reshape(3,))
#             lamA_v = csdl.Variable(value=lamA_np.reshape(1,))
#             lamB_v = csdl.Variable(value=lamB_np.reshape(1,))

#             # SDFs and gradients
#             phiA_a = phi_A(a_v); gA = csdl.reshape(csdl.derivative(ofs=phiA_a, wrts=a_v), (3,))
#             phiB_b = phi_B(b_v); gB = csdl.reshape(csdl.derivative(ofs=phiB_b, wrts=b_v), (3,))

#             diff = a_v - b_v

#             # Residuals:
#             # R1 = (a-b) + lamA * grad(phi_A)(a)    (3)
#             # R2 = -(a-b) + lamB * grad(phi_B)(b)   (3)
#             # R3 = w_phi * phi_A(a)                 (1)
#             # R4 = w_phi * phi_B(b)                 (1)
#             R1 = diff + csdl.reshape(lamA_v,(1,)) * gA
#             R2 = -diff + csdl.reshape(lamB_v,(1,)) * gB
#             R3 = w_phi * phiA_a
#             R4 = w_phi * phiB_b

#             F = np.zeros(8, float)
#             F[0:3] = _to_np(R1,(3,))
#             F[3:6] = _to_np(R2,(3,))
#             F[6]   = float(R3.value)
#             F[7]   = float(R4.value)

#             # Jacobian via AD row-by-row
#             J = np.zeros((8,8), float)

#             def fill_row(i, r_scalar):
#                 da = csdl.derivative(ofs=r_scalar, wrts=a_v)  # (3,)
#                 db = csdl.derivative(ofs=r_scalar, wrts=b_v)  # (3,)
#                 dA = csdl.derivative(ofs=r_scalar, wrts=lamA_v) # (1,)
#                 dB = csdl.derivative(ofs=r_scalar, wrts=lamB_v) # (1,)
#                 J[i,0:3] = _to_np(da,(3,))
#                 J[i,3:6] = _to_np(db,(3,))
#                 J[i,6]   = float(dA.value)
#                 J[i,7]   = float(dB.value)

#             # R1 rows
#             fill_row(0, R1[0]); fill_row(1, R1[1]); fill_row(2, R1[2])
#             # R2 rows
#             fill_row(3, R2[0]); fill_row(4, R2[1]); fill_row(5, R2[2])
#             # R3, R4 rows
#             fill_row(6, R3); fill_row(7, R4)

#             # Also return raw φ for taper decisions
#             return F, J, float(phiA_a.value), float(phiB_b.value)

#         F, J, phiA_val, phiB_val = FJ(x); f2 = float(F @ F)
#         for _ in range(max_iter):
#             JTJ = J.T @ J
#             g = J.T @ F
#             try:
#                 dx = -np.linalg.solve(JTJ + mu*np.eye(8), g)
#             except np.linalg.LinAlgError:
#                 mu *= lm_mu_inc; continue

#             x_trial = x + dx
#             F_trial, J_trial, phiA_t, phiB_t = FJ(x_trial)
#             f2_trial = float(F_trial @ F_trial)

#             # Predicted reduction (Gauss–Newton model)
#             pred_red = - (dx @ g) - 0.5 * (dx @ (JTJ @ dx))
#             rho = (f2 - f2_trial) / max(pred_red, 1e-16)

#             if rho > lm_eta and f2_trial < f2:
#                 x, F, J, f2 = x_trial, F_trial, J_trial, f2_trial
#                 phiA_val, phiB_val = phiA_t, phiB_t
#                 mu = max(mu * lm_mu_dec, 1e-15)
#                 if f2 < kkt_tol:
#                     break
#             else:
#                 mu *= lm_mu_inc

#         a_np, b_np, lamA_np, lamB_np = unpack(x)
#         return a_np, b_np, lamA_np, lamB_np, f2, max(abs(phiA_val), abs(phiB_val))

#     # ---------- Build Phase-1 seeds (two orders), run Phase-2 with schedule ----------
#     x0_np = _to_np(x0, (3,))

#     # Order A→B
#     a_seed_A = _lm_project(phi_A, x0_np)
#     b_seed_A = _lm_project(phi_B, a_seed_A)

#     # Order B→A
#     b_seed_B = _lm_project(phi_B, x0_np)
#     a_seed_B = _lm_project(phi_A, b_seed_B)

#     def run_schedule(a0_np, b0_np):
#         a_np, b_np = a0_np.copy(), b0_np.copy()
#         # Feasibility-weighted passes
#         w = max(1.0, float(feas_weight_init))
#         for s in range(max(1, feas_schedule_steps)):
#             a_np, b_np, lamA_np, lamB_np, f2, feas = _kkt_LM(
#                 a_np, b_np, w_phi=w, max_iter=kkt_max_iter, tol=kkt_tol
#             )
#             # Taper weight if feasible enough
#             if feas <= feas_taper_threshold:
#                 w = 1.0
#                 break
#             w = max(1.0, w * float(feas_weight_decay))
#         # Final refinement at weight=1 (ensures unbiased stationarity)
#         a_np, b_np, lamA_np, lamB_np, f2, feas = _kkt_LM(
#             a_np, b_np, w_phi=1.0, max_iter=kkt_max_iter, tol=kkt_tol
#         )
#         return a_np, b_np, f2, feas

#     aA_np, bA_np, f2A, feasA = run_schedule(a_seed_A, b_seed_A)
#     aB_np, bB_np, f2B, feasB = run_schedule(a_seed_B, b_seed_B)

#     # Pick the better candidate (prefer smaller gap; tie-break by feasibility & f2)
#     def gap_of(a_np, b_np):
#         diff = a_np - b_np
#         return float(np.sqrt(np.dot(diff, diff)))

#     gapA = gap_of(aA_np, bA_np)
#     gapB = gap_of(aB_np, bB_np)

#     if (gapA < gapB) or (abs(gapA-gapB) < 1e-12 and (feasA, f2A) <= (feasB, f2B)):
#         a_best, b_best, gap_best = aA_np, bA_np, gapA
#     else:
#         a_best, b_best, gap_best = aB_np, bB_np, gapB

#     # ---- Pack outputs as CSDL Variables (for downstream graph use) ----
#     a_v = csdl.Variable(value=a_best.reshape(3,))
#     b_v = csdl.Variable(value=b_best.reshape(3,))
#     diff_v = a_v - b_v
#     gap_v = csdl.sqrt(csdl.vdot(diff_v, diff_v) + 1e-16)
#     x_mid = 0.5 * (a_v + b_v)

#     if return_all:
#         return x_mid, a_v, b_v, gap_v
#     return gap_v
