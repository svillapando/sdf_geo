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
