import numpy as np
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except ImportError:
    pv = None


def plot_2d_slice(phi: np.ndarray, x: np.ndarray, y: np.ndarray, z_idx: int = None, title: str = "φ(x, y, z=0)"):
    """Plot a 2D Z-slice of a 3D SDF field using matplotlib."""
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
    """Render a 3D isosurface from a SDF using PyVista."""
    if pv is None:
        raise ImportError("PyVista is not installed. Run `pip install pyvista`.")

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
