"""
viz_poisson.py
==============
Reconstrucción de superficie de Poisson para comparar el tumor (ET) predicho
contra el ground-truth en 3D.

Pipeline:
  1.  máscara binaria  ->  nube de puntos orientada (vértices + normales) vía
      marching cubes sobre el campo de distancia con signo suavizado.
  2.  Poisson screened surface reconstruction (open3d) -> malla watertight.
  3.  Render comparativo GT (verde) vs predicción (rojo) y guardado a PNG.
      Si open3d no está disponible, cae a marching cubes + matplotlib 3D.

Se invoca solo para los casos/métodos que superan el umbral de Dice pedido
(por defecto 0.75), tal como solicitó el usuario.
"""
from __future__ import annotations
import os
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:                       # pragma: no cover
    _HAS_O3D = False


# --------------------------------------------------------------------------- #
# Nube de puntos orientada a partir de una máscara binaria
# --------------------------------------------------------------------------- #
def _mascara_a_nube(mask: np.ndarray, paso: float = 1.0):
    """Devuelve (puntos Nx3, normales Nx3) de la superficie de la máscara.

    Usa marching cubes sobre la SDF suavizada: los vértices están sobre la
    superficie y las normales se obtienen del gradiente de la SDF (orientadas
    hacia afuera), que es justo lo que la reconstrucción de Poisson necesita.
    """
    m = mask.astype(bool)
    if m.sum() < 10:
        return None, None
    # Distancia con signo: + fuera, - dentro -> el gradiente apunta hacia afuera.
    sdf = (ndimage.distance_transform_edt(~m)
           - ndimage.distance_transform_edt(m)).astype(np.float32)
    sdf = ndimage.gaussian_filter(sdf, 1.0)
    try:
        verts, faces, normals, _ = marching_cubes(sdf, level=0.0)
    except Exception:
        return None, None
    if paso != 1.0:
        verts = verts * paso
    return verts.astype(np.float64), normals.astype(np.float64)


def _poisson_mesh(mask: np.ndarray, depth: int = 8):
    """Malla de Poisson (open3d) a partir de la máscara, o None si falla."""
    if not _HAS_O3D:
        return None
    pts, nrm = _mascara_a_nube(mask)
    if pts is None:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False)
    # Recortar artefactos de baja densidad (típico del Poisson screened).
    dens = np.asarray(dens)
    if dens.size:
        thr = np.quantile(dens, 0.02)
        mesh.remove_vertices_by_mask(dens < thr)
    mesh.compute_vertex_normals()
    return mesh


# --------------------------------------------------------------------------- #
# Render comparativo GT vs predicción
# --------------------------------------------------------------------------- #
def comparar_superficies(gt_mask: np.ndarray,
                         pred_mask: np.ndarray,
                         case_id: str,
                         metodo: str,
                         dice_val: float,
                         path_out: str,
                         exportar_ply: bool = True) -> bool:
    """
    Genera la figura comparativa de superficies de Poisson (GT vs pred).
    Devuelve True si se generó con open3d, False si usó el fallback matplotlib.
    """
    os.makedirs(os.path.dirname(path_out), exist_ok=True)

    if _HAS_O3D:
        ok = _render_open3d(gt_mask, pred_mask, case_id, metodo,
                            dice_val, path_out, exportar_ply)
        if ok:
            return True
    _render_matplotlib(gt_mask, pred_mask, case_id, metodo, dice_val, path_out)
    return False


def _anotar_png(path, case_id, metodo, dice_val):
    """Añade barra de título (caso · método · Dice, GT verde / pred rojo) al PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100 + 0.7))
        ax = fig.add_axes([0, 0, 1, 0.92]); ax.imshow(img); ax.axis("off")
        fig.suptitle(
            f"{case_id} — Superficie de Poisson | {metodo} | Dice={dice_val:.3f}\n"
            f"GT (verde, izq.)   vs   predicción (rojo, der.)",
            fontsize=11, y=0.99)
        fig.savefig(path, dpi=100)
        plt.close(fig)
    except Exception:
        pass


def _render_open3d(gt_mask, pred_mask, case_id, metodo, dice_val,
                   path_out, exportar_ply) -> bool:
    try:
        mesh_gt = _poisson_mesh(gt_mask)
        mesh_pr = _poisson_mesh(pred_mask)
        if mesh_gt is None or mesh_pr is None:
            return False
        mesh_gt.paint_uniform_color([0.20, 0.75, 0.30])   # verde = GT
        mesh_pr.paint_uniform_color([0.85, 0.20, 0.20])   # rojo  = pred

        if exportar_ply:
            base = os.path.splitext(path_out)[0]
            o3d.io.write_triangle_mesh(base + "_GT.ply", mesh_gt)
            o3d.io.write_triangle_mesh(base + "_pred.ply", mesh_pr)

        # Render headless (offscreen) a imagen.
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1200, height=600)
            # GT desplazado a la izquierda, pred a la derecha para comparar.
            gt_shift = o3d.geometry.TriangleMesh(mesh_gt)
            ext = mesh_gt.get_axis_aligned_bounding_box().get_extent()[0]
            gt_shift.translate((-ext * 0.7, 0, 0))
            pr_shift = o3d.geometry.TriangleMesh(mesh_pr)
            pr_shift.translate((ext * 0.7, 0, 0))
            vis.add_geometry(gt_shift)
            vis.add_geometry(pr_shift)
            opt = vis.get_render_option()
            opt.background_color = np.array([1.0, 1.0, 1.0])
            vis.poll_events(); vis.update_renderer()
            vis.capture_screen_image(path_out, do_render=True)
            vis.destroy_window()
            if os.path.exists(path_out):
                _anotar_png(path_out, case_id, metodo, dice_val)
                return True
            return False
        except Exception:
            # Sin contexto GL: render overlay con matplotlib pero malla Poisson.
            _render_meshes_matplotlib(mesh_gt, mesh_pr, case_id, metodo,
                                      dice_val, path_out)
            return True
    except Exception:
        return False


def _render_meshes_matplotlib(mesh_gt, mesh_pr, case_id, metodo, dice_val,
                              path_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 6))
    for i, (mesh, titulo, col) in enumerate([
            (mesh_gt, "Ground truth", (0.20, 0.75, 0.30)),
            (mesh_pr, f"Predicción ({metodo})", (0.85, 0.20, 0.20))]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        if len(f):
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolor=col, edgecolor="none", alpha=0.9))
            ax.set_xlim(v[:, 0].min(), v[:, 0].max())
            ax.set_ylim(v[:, 1].min(), v[:, 1].max())
            ax.set_zlim(v[:, 2].min(), v[:, 2].max())
        ax.set_title(titulo); ax.set_axis_off()
    fig.suptitle(f"{case_id} — Poisson surface | {metodo} | Dice={dice_val:.3f}")
    fig.tight_layout()
    fig.savefig(path_out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _render_matplotlib(gt_mask, pred_mask, case_id, metodo, dice_val, path_out):
    """Fallback puro: marching cubes + matplotlib (sin open3d)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 6))
    for i, (mask, titulo, col) in enumerate([
            (gt_mask, "Ground truth", (0.20, 0.75, 0.30)),
            (pred_mask, f"Predicción ({metodo})", (0.85, 0.20, 0.20))]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        m = mask.astype(bool)
        if m.sum() > 20:
            sdf = ndimage.gaussian_filter(
                (ndimage.distance_transform_edt(~m)
                 - ndimage.distance_transform_edt(m)).astype(np.float32), 1.0)
            try:
                v, f, _, _ = marching_cubes(sdf, level=0.0)
                ax.add_collection3d(Poly3DCollection(
                    v[f], facecolor=col, edgecolor="none", alpha=0.9))
                ax.set_xlim(v[:, 0].min(), v[:, 0].max())
                ax.set_ylim(v[:, 1].min(), v[:, 1].max())
                ax.set_zlim(v[:, 2].min(), v[:, 2].max())
            except Exception:
                pass
        ax.set_title(titulo); ax.set_axis_off()
    fig.suptitle(f"{case_id} — Poisson surface | {metodo} | Dice={dice_val:.3f}")
    fig.tight_layout()
    fig.savefig(path_out, dpi=110, bbox_inches="tight")
    plt.close(fig)
