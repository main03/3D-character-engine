from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np
import trimesh
from PIL import Image

ProgressFn = Callable[[float], None]


@dataclass
class Reconstruction:
    """Raw output of an image-to-3D model.

    `mesh` must be in the canonical frame: Y-up, character facing +Z, metres-agnostic.
    `color_fn` maps (N, 3) canonical-frame points to (N, 3) RGB in [0, 1]; when given, the
    texture stage samples it directly instead of interpolating vertex colours.
    """

    mesh: trimesh.Trimesh
    color_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


class Reconstructor(Protocol):
    name: str

    def reconstruct(self, image: Image.Image, progress: ProgressFn) -> Reconstruction:
        """Build a coloured mesh from a framed RGB image."""
        ...
