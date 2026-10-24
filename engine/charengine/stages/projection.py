"""Project the input photo onto the camera-facing surface of the reconstruction.

Image-to-3D models see the photo at low resolution (TripoSR: 512px for the whole subject), so
facial detail in their predicted colours is blurry. The photo itself is the best source of
colour for every surface point the camera actually saw.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

from ..reconstruct.base import ViewCamera
from .raster import rasterize


def _project(points: np.ndarray, camera: ViewCamera, size: int):
    """Canonical points -> (pixel x, pixel y, depth) in a size x size image of the input view."""
    f = 1.0 / np.tan(np.radians(camera.fov_deg) / 2.0)
    depth = camera.distance - points[:, 2]
    x = f * points[:, 0] / depth
    y = f * points[:, 1] / depth
    return (x + 1.0) / 2.0 * size - 0.5, (1.0 - y) / 2.0 * size - 0.5, depth


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bilinear(image: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (px - x0)[:, None]
    fy = (py - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def depth_buffer(vertices: np.ndarray, faces: np.ndarray, camera: ViewCamera, size: int) -> np.ndarray:
    px, py, depth = _project(vertices, camera, size)
    tri = np.stack([px, py], axis=1)[faces]
    ys, xs, fids, bary = rasterize(tri, size, size)
    frag_depth = np.einsum("nk,nk->n", bary, depth[faces[fids]])
    buf = np.full((size, size), np.inf)
    np.minimum.at(buf, (ys, xs), frag_depth)
    return buf


def project_photo(
    points: np.ndarray,
    normals: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: ViewCamera,
    photo: Image.Image,
    depth_resolution: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (colours (N, 3) in [0, 1], blend weights (N,) in [0, 1]) for surface points."""
    rgba = np.asarray(photo.convert("RGBA")).astype(np.float32) / 255.0
    size = rgba.shape[0]

    # Visibility: the point must be the front-most surface along its camera ray.
    zbuf = depth_buffer(vertices, faces, camera, depth_resolution)
    zbuf = ndimage.minimum_filter(zbuf, size=3)  # be lenient at triangle boundaries
    dx, dy, d = _project(points, camera, depth_resolution)
    ix = np.clip(np.round(dx).astype(np.int64), 0, depth_resolution - 1)
    iy = np.clip(np.round(dy).astype(np.int64), 0, depth_resolution - 1)
    visible = d <= zbuf[iy, ix] + 0.015

    # Facing: full weight when the surface faces the camera, fading out towards grazing angles,
    # where the photo is stretched and the model's own colour is more reliable.
    to_camera = np.array([0.0, 0.0, camera.distance]) - points
    to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True)
    facing = np.einsum("nd,nd->n", normals, to_camera)

    # Stay inside the subject mask, eroded slightly so background fringe never lands on the model.
    erode = max(1, size // 150)
    alpha = ndimage.minimum_filter(rgba[..., 3], size=2 * erode + 1)

    px, py, _ = _project(points, camera, size)
    colors = _bilinear(rgba[..., :3], px, py)
    inside = _bilinear(alpha[..., None], px, py)[:, 0]
    weights = visible * _smoothstep(0.2, 0.55, facing) * _smoothstep(0.5, 0.95, inside)
    return colors, weights


def match_colors(base: np.ndarray, photo: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Shift the model's predicted colours toward the photo's statistics where both are known.

    Keeps the unseen sides and back consistent with the projected front (skin tone, exposure).
    """
    sel = weights > 0.8
    if sel.sum() < 500:
        return base
    mb, sb = base[sel].mean(0), base[sel].std(0) + 1e-6
    mp, sp = photo[sel].mean(0), photo[sel].std(0) + 1e-6
    scale = np.clip(sp / sb, 0.7, 1.5)
    return np.clip((base - mb) * scale + mp, 0.0, 1.0)
