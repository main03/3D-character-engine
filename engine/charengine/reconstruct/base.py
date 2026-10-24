from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np
import trimesh
from PIL import Image

ProgressFn = Callable[[float], None]


@dataclass(frozen=True)
class ViewCamera:
    """Perspective camera of the input photo in the canonical frame: at (0, 0, distance) looking at -Z,
    Y up, square image spanning the framed photo."""

    distance: float
    fov_deg: float


@dataclass
class Reconstruction:
    """Raw output of an image-to-3D model.

    `mesh` must be in the canonical frame: Y-up, character facing +Z, metres-agnostic.
    `color_fn` maps (N, 3) canonical-frame points to (N, 3) RGB in [0, 1]; when given, the
    texture stage samples it directly instead of interpolating vertex colours.
    `view`, when known, lets the texture stage project the original photo onto visible surfaces.
    """

    mesh: trimesh.Trimesh
    color_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    view: Optional[ViewCamera] = None


class Reconstructor(Protocol):
    name: str

    def reconstruct(self, image: Image.Image, progress: ProgressFn) -> Reconstruction:
        """Build a coloured mesh from a framed RGB image."""
        ...
