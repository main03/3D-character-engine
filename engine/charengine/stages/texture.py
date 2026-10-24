"""UV unwrapping and colour baking into a base-colour texture (CPU only, no OpenGL)."""

from dataclasses import dataclass

import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy import ndimage

from ..config import PipelineConfig


@dataclass
class TexturedMesh:
    vertices: np.ndarray  # (V, 3), split along UV seams
    faces: np.ndarray  # (F, 3)
    normals: np.ndarray  # (V, 3), from the welded mesh so seams shade smoothly
    uvs: np.ndarray  # (V, 2), OpenGL convention (v up); the GLB exporter flips for glTF
    texture: Image.Image


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


def rasterize_uv(faces: np.ndarray, uvs: np.ndarray, size: int, chunk: int = 4096):
    """For each texel covered by a UV triangle, return its face index and barycentric weights.

    Returns (texel_ys, texel_xs, face_ids, bary) with bary of shape (N, 3).
    """
    px = np.empty_like(uvs)
    px[:, 0] = uvs[:, 0] * size - 0.5
    px[:, 1] = (1.0 - uvs[:, 1]) * size - 0.5  # image row 0 is v = 1
    tri = px[faces]  # (F, 3, 2)

    out = ([], [], [], [])
    for start in range(0, len(faces), chunk):
        t = tri[start : start + chunk]
        lo = np.clip(np.floor(t.min(axis=1)), 0, size - 1).astype(np.int64)
        hi = np.clip(np.ceil(t.max(axis=1)), 0, size - 1).astype(np.int64)
        w = hi[:, 0] - lo[:, 0] + 1
        h = hi[:, 1] - lo[:, 1] + 1
        counts = w * h
        fid = np.repeat(np.arange(len(t)), counts)
        local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        xs = lo[fid, 0] + local % w[fid]
        ys = lo[fid, 1] + local // w[fid]

        a, b, c = t[fid, 0], t[fid, 1], t[fid, 2]
        v0, v1 = b - a, c - a
        v2 = np.stack([xs, ys], axis=1) - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        valid = np.abs(den) > 1e-12
        den = np.where(valid, den, 1.0)
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        w0 = 1.0 - w1 - w2
        eps = -1e-4
        inside = valid & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)

        out[0].append(ys[inside])
        out[1].append(xs[inside])
        out[2].append(fid[inside] + start)
        out[3].append(np.stack([w0, w1, w2], axis=1)[inside])
    return tuple(np.concatenate(o) for o in out)


def bake_texture(mesh: trimesh.Trimesh, color_fn, config: PipelineConfig) -> TexturedMesh:
    size = config.texture_size
    vmapping, faces, uvs = unwrap(mesh, size)
    vertices = mesh.vertices[vmapping]
    normals = mesh.vertex_normals[vmapping]

    ys, xs, fids, bary = rasterize_uv(faces, uvs, size)
    corners = faces[fids]  # (N, 3)
    if color_fn is not None:
        points = np.einsum("nk,nkd->nd", bary, vertices[corners])
        colors = color_fn(points)
    else:
        vc = mesh.visual.vertex_colors[vmapping, :3].astype(np.float32) / 255.0
        colors = np.einsum("nk,nkd->nd", bary, vc[corners])

    tex = np.zeros((size, size, 3), dtype=np.float32)
    covered = np.zeros((size, size), dtype=bool)
    tex[ys, xs] = np.clip(colors, 0.0, 1.0)
    covered[ys, xs] = True
    # Fill empty texels with the nearest baked colour so bilinear filtering does not bleed black at seams.
    if covered.any():
        idx = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
        tex = tex[idx[0], idx[1]]

    image = Image.fromarray((tex * 255.0).round().astype(np.uint8))
    return TexturedMesh(vertices=vertices, faces=faces, normals=normals, uvs=uvs, texture=image)
