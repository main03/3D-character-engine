"""Geometry clean-up: drop floaters, decimate, normalise placement."""

import numpy as np
import trimesh

from ..config import PipelineConfig
from ..errors import ReconstructionError


def clean_mesh(mesh: trimesh.Trimesh, config: PipelineConfig) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(len(p.faces) for p in parts)
        keep = [p for p in parts if len(p.faces) >= config.min_component_ratio * largest]
        mesh = trimesh.util.concatenate(keep)

    if config.smoothing_iterations:
        # Taubin smoothing removes marching-cubes stair-stepping without shrinking the volume.
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=config.smoothing_iterations)

    if len(mesh.faces) > config.target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=config.target_faces)

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < 100:
        raise ReconstructionError("Reconstructed geometry is too small to be a usable character.")
    return mesh


def placement_transform(mesh: trimesh.Trimesh, config: PipelineConfig) -> np.ndarray:
    """Transform that stands the character on y=0, centred on x/z, scaled to target_height."""
    lo, hi = mesh.bounds
    height = hi[1] - lo[1]
    if not np.isfinite(height) or height <= 0:
        raise ReconstructionError("Reconstructed geometry has invalid bounds.")
    scale = config.target_height / height
    center = (lo + hi) / 2
    t = np.eye(4)
    t[:3, :3] *= scale
    t[:3, 3] = -scale * np.array([center[0], lo[1], center[2]])
    return t
