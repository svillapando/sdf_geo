import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

def plot_2d_slice(phi: np.ndarray, x: np.ndarray, y: np.ndarray, z_idx: int = None, title: str = "φ(x, y, z=0)"):
    #Plot a 2D Z-slice of a 3D SDF field using matplotlib
    if z_idx is None:
        z_idx = phi.shape[2] // 2  # middle slice by default
    phi_slice = phi[:, :, z_idx]
    plt.figure()
    cf = plt.contourf(x, y, phi_slice.T, levels=50, cmap='RdBu')
    plt.contour(x, y, phi_slice.T, levels=[0.0], colors='black', linewidths=2)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(cf, label='Signed Distance')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_3d_isosurface(phi: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, level: float = 0.0, title: str = "Isosurface φ = 0"):
    # Render a 3D isosurface from a SDF using PyVista
    
    grid = pv.ImageData()
    grid.dimensions = np.array(phi.shape)
    grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    grid.origin = (x[0], y[0], z[0])
    grid.point_data["phi"] = phi.flatten(order="F")

    contour = grid.contour(isosurfaces=[level], scalars="phi")

    plotter = pv.Plotter()
    plotter.add_mesh(contour, color="blue", opacity=0.7)
    plotter.add_axes()
    plotter.add_title(title)
    plotter.show()


def plot_isosurface_with_collision_points(
    phi, x, y, z, *,
    level=0.0,
    title="Drone  Ball Collision Check",
    xstar=None,             # (3,) or (K,3)
    p_near_drone=None,      # (3,) or (K,3)
    p_near_env=None,        # (3,) or (K,3)
    seed_points=None,       # (M,3)
    point_px=6,             # visual size of the dots (pixels)
    label_font_size=14,
    label_offset_px=(8, 8), # offset of text in screen space
    show_labels=True,
    show_axes=True,
    show_legend=False,      # legend disabled (prevents overlap)
):
    """
    Render φ(x,y,z)=level and overlay very small markers + optional 2D labels
    (no leader lines; labels are screen‑space so they can’t intersect geometry).
    """
    if pv is None:
        raise ImportError("PyVista is not installed. Run `pip install pyvista`.")

    # --- grid & isosurface ---
    grid = pv.ImageData()
    grid.dimensions = np.array(phi.shape)
    grid.spacing = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
    grid.origin = (x[0], y[0], z[0])
    grid.point_data["phi"] = phi.flatten(order="F")
    contour = grid.contour(isosurfaces=[level], scalars="phi")

    plotter = pv.Plotter(window_size=(1200, 900))
    plotter.add_mesh(contour, color="royalblue", opacity=0.7, name="isosurface")
    if show_axes:
        plotter.add_axes()
    plotter.add_text(title, position="upper_edge", font_size=22, color="black", shadow=True)

    # --- helpers ---
    def _asNx3(a):
        if a is None: return None
        a = np.asarray(a, float)
        return a.reshape(1, 3) if a.ndim == 1 else a

    def _add_points(pts, color, name):
        if pts is None or len(pts) == 0: return
        plotter.add_points(
            pts, color=color, point_size=point_px,
            render_points_as_spheres=True, name=name
        )

    def _add_labels(pts, text, color):
        if pts is None or len(pts) == 0: return
        # text can be a single string or list of strings (len == len(pts))
        if isinstance(text, str):
            texts = [text] if len(pts) == 1 else [f"{text} {i}" for i in range(len(pts))]
        else:
            texts = text
        plotter.add_point_labels(
            pts, texts, font_size=label_font_size, text_color=color,
            point_size=1, show_points=False, always_visible=True,
            shape=None, # no box
            # screen‑space offsets (pixels):
            # PyVista doesn’t expose an explicit pixel offset; emulate by
            # adding a small world offset in view direction after first render.
        )

    # data
    Xs   = _asNx3(xstar)
    Pd   = _asNx3(p_near_drone)
    Pe   = _asNx3(p_near_env)
    Seed = _asNx3(seed_points)

    colors = {
        "x* (min F)":        "magenta",
        "nearest on drone":  "orange",
        "nearest on env":    "limegreen",
        "seed":              "gray",
    }

    # tiny markers
    _add_points(Xs,   colors["x* (min F)"],       "xstar")
    _add_points(Pd,   colors["nearest on drone"], "p_drone")
    _add_points(Pe,   colors["nearest on env"],   "p_env")
    _add_points(Seed, colors["seed"],             "seed")

    # optional small labels (no leader lines)
    if show_labels:
        _add_labels(Xs,   "x*",               colors["x* (min F)"])
        _add_labels(Pd,   "nearest on drone", colors["nearest on drone"])
        _add_labels(Pe,   "nearest on env",   colors["nearest on env"])
        # usually omit labels for seeds to reduce clutter
        # _add_labels(Seed, "seed",             colors["seed"])

    if show_legend:
        plotter.add_legend([
            ["x* (min F)",         colors["x* (min F)"]],
            ["nearest on A",   colors["nearest on drone"]],
            ["nearest on B",     colors["nearest on env"]],
            ["seed",               colors["seed"]],
        ])

    plotter.show()



def plot_3d_eikonal_violation(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    absolute: bool = True,          # plot |‖∇φ‖-1| (True) or signed (‖∇φ‖-1)
    as_volume: bool = True,         # volume rendering (True) or isosurfaces (False)
    iso_levels = (0.05, 0.10, 0.20),# isosurface levels for violation (in absolute units)
    title: str = "Eikonal violation map",
    cmap: str = "inferno",
    opacity: float | None = None,   # None: auto ramp for volume
    edge_order: int = 2,             # 2nd-order finite diff at edges
    cool: bool = False              # Clear screen for screenshots
):
    """
    Plot a 3D map of the eikonal violation from a gridded SDF φ(x,y,z).

    Parameters
    ----------
    phi : (nx, ny, nz) array
        SDF samples on a rectilinear grid defined by x, y, z.
    x, y, z : 1D arrays
        Grid axes (monotone). Uniform spacing assumed for visualization.
    absolute : bool
        If True, visualize |‖∇φ‖ - 1|; otherwise visualize ‖∇φ‖ - 1 (signed).
    as_volume : bool
        If True, use volume rendering; else show isosurfaces of violation.
    iso_levels : tuple[float,...]
        Violation magnitudes to contour when as_volume=False.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name for coloring.
    opacity : float | None
        Global mesh opacity for isosurfaces; None lets PyVista decide for volume.
    edge_order : int
        Order for np.gradient at edges (1 or 2).

    Returns
    -------
    grad_norm : np.ndarray
        The computed ‖∇φ‖ on the same grid.
    violation : np.ndarray
        Either |‖∇φ‖-1| (absolute=True) or (‖∇φ‖-1) (signed).
    """
    # --- finite-difference gradient on the rectilinear grid ---
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])

    gx, gy, gz = np.gradient(phi, dx, dy, dz, edge_order=edge_order)
    grad_norm = np.sqrt(gx*gx + gy*gy + gz*gz)

    # eikonal violation
    if absolute:
        violation = np.abs(grad_norm - 1.0)
        scalars_name = "eikonal_abs"
    else:
        violation = grad_norm - 1.0
        scalars_name = "eikonal_signed"

    # --- build a PyVista uniform grid matching your layout ---
    grid = pv.ImageData()
    grid.dimensions = np.array(phi.shape)            # (nx, ny, nz)
    grid.spacing = (dx, dy, dz)
    grid.origin = (x[0], y[0], z[0])

    # Important: keep Fortran order to match pv.ImageData memory expectations
    grid.point_data[scalars_name] = violation.flatten(order="F")

    # Optional: also attach grad_norm if you want to inspect later
    grid.point_data["grad_norm"] = grad_norm.flatten(order="F")

    if cool == True:
        scale = False
    else:
        scale = True

    # --- visualize ---
    plotter = pv.Plotter()
    if as_volume:
        # Volume render the violation; smaller values -> more transparent
        # Build a simple opacity transfer if none provided
        if opacity is None:
            # Build a simple opacity ramp: transparent at 0, opaque by ~max violation
            vmax = np.percentile(violation, 99.5)
            vmax = vmax if vmax > 0 else 1.0
            opacity = [0.0, 0.2, 0.6, 1.0]   # piecewise values
        plotter.add_volume(grid, scalars=scalars_name, cmap=cmap, opacity=opacity)
    else:
        # Contour specific violation magnitudes (isosurfaces)
        # Use positive magnitudes even when signed=False
        levels = np.asarray(iso_levels, float)
        levels = np.unique(np.abs(levels))
        if levels.size == 0:
            levels = np.array([0.1], float)
        contour = grid.contour(isosurfaces=list(levels), scalars=scalars_name)
        plotter.add_mesh(contour, cmap=cmap, opacity=opacity if opacity is not None else 0.6, show_scalar_bar = scale)

    if cool == False:
        plotter.add_axes()
        plotter.add_title(title)
        plotter.show()
    else:
        plotter.show()

    return grad_norm, violation