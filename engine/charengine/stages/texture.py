"""UV unwrapping and colour baking into a base-colour texture (CPU only, no OpenGL)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy import ndimage

from ..config import PipelineConfig
from ..reconstruct.base import Reconstruction
from . import projection
from .raster import rasterize


@dataclass
class TexturedMesh:
    vertices: np.ndarray  # (V, 3), split along UV seams
    faces: np.ndarray  # (F, 3)
    normals: np.ndarray  # (V, 3), from the welded mesh so seams shade smoothly
    uvs: np.ndarray  # (V, 2), OpenGL convention (v up); the GLB exporter flips for glTF
    texture: Image.Image
    photo_coverage: float = 0.0  # fraction of baked texels coloured mostly from the photo


def unwrap(mesh: trimesh.Trimesh, size: int):
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = size
    pack.padding = max(2, size // 256)
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    vmapping, indices, uvs = atlas[0]
    return vmapping.astype(np.int64), indices.astype(np.int64), uvs.astype(np.float64)


def rasterize_uv(faces: np.ndarray, uvs: np.ndarray, size: int):
    """For each texel covered by a UV triangle, return (ys, xs, face_ids, barycentric weights)."""
    px = np.empty_like(uvs)
    px[:, 0] = uvs[:, 0] * size - 0.5
    px[:, 1] = (1.0 - uvs[:, 1]) * size - 0.5  # image row 0 is v = 1
    return rasterize(px[faces], size, size)


def bake_texture(
    mesh: trimesh.Trimesh,
    recon: Reconstruction,
    config: PipelineConfig,
    photo: Optional[Image.Image] = None,
) -> TexturedMesh:
    size = config.texture_size
    vmapping, faces, uvs = unwrap(mesh, size)
    vertices = mesh.vertices[vmapping]
    normals = mesh.vertex_normals[vmapping]

    ys, xs, fids, bary = rasterize_uv(faces, uvs, size)
    corners = faces[fids]  # (N, 3)
    points = np.einsum("nk,nkd->nd", bary, vertices[corners])
    if recon.color_fn is not None:
        colors = recon.color_fn(points)
    else:
        vc = mesh.visual.vertex_colors[vmapping, :3].astype(np.float32) / 255.0
        colors = np.einsum("nk,nkd->nd", bary, vc[corners])

    coverage = 0.0
    if config.project_photo and photo is not None and recon.view is not None:
        texel_normals = np.einsum("nk,nkd->nd", bary, normals[corners])
        texel_normals /= np.linalg.norm(texel_normals, axis=1, keepdims=True).clip(1e-12)
        photo_rgb, weights = projection.project_photo(
            points, texel_normals, mesh.vertices, mesh.faces, recon.view, photo
        )
        colors = projection.match_colors(colors, photo_rgb, weights)
        colors = colors * (1.0 - weights[:, None]) + photo_rgb * weights[:, None]
        coverage = float((weights > 0.5).mean()) if len(weights) else 0.0

    tex = np.zeros((size, size, 3), dtype=np.float32)
    covered = np.zeros((size, size), dtype=bool)
    tex[ys, xs] = np.clip(colors, 0.0, 1.0)
    covered[ys, xs] = True
    # Fill empty texels with the nearest baked colour so bilinear filtering does not bleed black at seams.
    if covered.any():
        idx = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
        tex = tex[idx[0], idx[1]]

    image = Image.fromarray((tex * 255.0).round().astype(np.uint8))
    return TexturedMesh(vertices, faces, normals, uvs, image, coverage)
