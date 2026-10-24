"""Structural validation of generated GLB files."""

import io
from dataclasses import dataclass, field

import numpy as np
import pygltflib
import trimesh

REQUIRED_ATTRIBUTES = ("POSITION", "NORMAL", "TEXCOORD_0")


@dataclass
class GlbReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_glb(data: bytes, max_bytes: int) -> GlbReport:
    errors: list[str] = []
    stats: dict = {"bytes": len(data)}

    if len(data) > max_bytes:
        errors.append(f"GLB is {len(data)} bytes, over the {max_bytes} byte limit.")
    if len(data) < 12 or data[:4] != b"glTF" or int.from_bytes(data[4:8], "little") != 2:
        return GlbReport(False, errors + ["Missing or unsupported glTF 2.0 binary header."], stats)
    if int.from_bytes(data[8:12], "little") != len(data):
        return GlbReport(False, errors + ["GLB header length does not match file size."], stats)

    try:
        gltf = pygltflib.GLTF2.load_from_bytes(data)
    except Exception as exc:  # pygltflib raises assorted exception types on malformed input
        return GlbReport(False, errors + [f"Could not parse GLB: {exc}"], stats)
    if gltf is None:
        return GlbReport(False, errors + ["Could not parse GLB."], stats)

    if gltf.asset.version != "2.0":
        errors.append(f"Unexpected glTF version {gltf.asset.version!r}.")
    if not gltf.meshes:
        errors.append("GLB contains no meshes.")
    for mi, mesh in enumerate(gltf.meshes):
        for pi, prim in enumerate(mesh.primitives):
            where = f"mesh {mi} primitive {pi}"
            attrs = prim.attributes
            for name in REQUIRED_ATTRIBUTES:
                if getattr(attrs, name, None) is None:
                    errors.append(f"{where} is missing {name}.")
            if prim.indices is None:
                errors.append(f"{where} has no index buffer.")
            if attrs.POSITION is not None:
                acc = gltf.accessors[attrs.POSITION]
                if not acc.min or not acc.max or not np.all(np.isfinite(acc.min + acc.max)):
                    errors.append(f"{where} has missing or non-finite POSITION bounds.")
                counts = {getattr(gltf.accessors[getattr(attrs, n)], "count") for n in REQUIRED_ATTRIBUTES if getattr(attrs, n, None) is not None}
                if len(counts) > 1:
                    errors.append(f"{where} has mismatched vertex attribute counts.")
            if prim.material is None:
                errors.append(f"{where} has no material.")
            else:
                pbr = gltf.materials[prim.material].pbrMetallicRoughness
                if pbr is None or pbr.baseColorTexture is None:
                    errors.append(f"{where} material has no base colour texture.")
    if not gltf.images:
        errors.append("GLB contains no embedded texture image.")
    for image in gltf.images:
        if image.bufferView is None or image.mimeType not in ("image/png", "image/jpeg"):
            errors.append("Texture images must be embedded PNG or JPEG.")

    try:
        loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
        stats.update(
            vertices=int(len(loaded.vertices)),
            triangles=int(len(loaded.faces)),
            extents=[round(float(x), 4) for x in loaded.extents],
        )
        if len(loaded.faces) < 100:
            errors.append("Mesh has too few triangles.")
        if not np.all(np.isfinite(loaded.vertices)):
            errors.append("Mesh has non-finite vertex positions.")
    except Exception as exc:
        errors.append(f"Could not reload GLB: {exc}")

    return GlbReport(not errors, errors, stats)
