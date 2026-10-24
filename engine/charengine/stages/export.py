"""GLB export."""

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from ..config import PipelineConfig
from .texture import TexturedMesh


def export_glb(textured: TexturedMesh, transform: np.ndarray, config: PipelineConfig) -> bytes:
    material = PBRMaterial(
        name="character",
        baseColorTexture=textured.texture,
        metallicFactor=0.0,
        roughnessFactor=config.roughness,
    )
    vertices = trimesh.transformations.transform_points(textured.vertices, transform)
    normals = textured.normals @ transform[:3, :3].T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=textured.faces,
        vertex_normals=normals,
        visual=TextureVisuals(uv=textured.uvs, material=material),
        process=False,
    )
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="character", geom_name="character")
    return trimesh.exchange.gltf.export_glb(scene, include_normals=True)
