import io

import numpy as np
import trimesh
from PIL import Image

from charengine.reconstruct import Reconstruction, ViewCamera


class SphereReconstructor:
    """Test double: a coloured sphere. Never used outside tests."""

    name = "test-sphere"

    def reconstruct(self, image, progress):
        progress(1.0)
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
        return Reconstruction(mesh=mesh, color_fn=lambda p: np.clip(p + 0.5, 0, 1), view=ViewCamera(1.8, 40.0))


def cutout(rgba: Image.Image) -> Image.Image:
    return rgba.convert("RGBA")


def png_bytes(size=(256, 256), mode="RGB", fmt="PNG") -> bytes:
    img = Image.new(mode, size, (200, 150, 120) if mode == "RGB" else (200, 150, 120, 255))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
