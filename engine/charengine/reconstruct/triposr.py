"""TripoSR backend (MIT, https://github.com/VAST-AI-Research/TripoSR), vendored in charengine.vendor.tsr."""

import threading

import numpy as np
import torch
import trimesh
from PIL import Image

from ..config import PipelineConfig
from ..errors import ReconstructionError
from .base import ProgressFn, Reconstruction, ViewCamera

# TripoSR is Z-up and faces -X; rotate to glTF's Y-up, facing +Z.
_TO_CANONICAL = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]) @ trimesh.transformations.rotation_matrix(
    -np.pi / 2, [1, 0, 0]
)
_FROM_CANONICAL = np.linalg.inv(_TO_CANONICAL)

# Input-view camera, fitted by matching projected mesh silhouettes to input masks (IoU ~0.97).
INPUT_VIEW = ViewCamera(distance=1.8, fov_deg=40.0)


class TripoSRReconstructor:
    name = "triposr"

    def __init__(self, config: PipelineConfig, device: str | None = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is None:
                from ..vendor.tsr.system import TSR

                model = TSR.from_pretrained(self.config.triposr_repo, config_name="config.yaml", weight_name="model.ckpt")
                model.renderer.set_chunk_size(self.config.chunk_size)
                self._model = model.to(self.device).eval()
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def reconstruct(self, image: Image.Image, progress: ProgressFn) -> Reconstruction:
        model = self.load()
        progress(0.1)
        with torch.no_grad():
            scene_codes = model([image], device=self.device)
        progress(0.4)
        mesh = model.extract_mesh(
            scene_codes, has_vertex_color=False, resolution=self.config.mc_resolution, threshold=self.config.mc_threshold
        )[0]
        progress(0.9)
        if len(mesh.faces) == 0:
            raise ReconstructionError("The model did not produce any geometry for this image.")
        mesh.apply_transform(_TO_CANONICAL)

        scene_code = scene_codes[0]

        def color_fn(points: np.ndarray) -> np.ndarray:
            local = trimesh.transformations.transform_points(points, _FROM_CANONICAL).astype(np.float32)
            out = np.empty((len(local), 3), dtype=np.float32)
            step = 1 << 16
            with torch.no_grad():
                for i in range(0, len(local), step):
                    pts = torch.from_numpy(local[i : i + step]).to(self.device)
                    out[i : i + step] = model.renderer.query_triplane(model.decoder, pts, scene_code)["color"].cpu().numpy()
            return out

        return Reconstruction(mesh=mesh, color_fn=color_fn, view=INPUT_VIEW)
