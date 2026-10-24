"""End-to-end orchestration: image bytes -> validated GLB bytes."""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import PipelineConfig
from .errors import ExportValidationError
from .reconstruct.base import Reconstructor
from .stages import export, mesh as mesh_stage, preprocess, texture, validate

# (stage name, start, end) as fractions of total progress
STAGES = {
    "validating_input": (0.00, 0.02),
    "removing_background": (0.02, 0.10),
    "reconstructing": (0.10, 0.70),
    "processing_mesh": (0.70, 0.78),
    "texturing": (0.78, 0.95),
    "exporting": (0.95, 1.00),
}

ProgressCallback = Callable[[str, float], None]


@dataclass
class GenerationResult:
    glb: bytes
    reconstructor: str
    stats: dict
    timings: dict
    warnings: list[str] = field(default_factory=list)


class CharacterPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        reconstructor: Optional[Reconstructor] = None,
        background_remover: Optional[Callable] = None,
    ):
        self.config = config or PipelineConfig()
        if reconstructor is None:
            from .reconstruct.triposr import TripoSRReconstructor

            reconstructor = TripoSRReconstructor(self.config)
        self.reconstructor = reconstructor
        self.background_remover = background_remover or preprocess.BackgroundRemover(self.config.background_model)

    def validate_input(self, data: bytes):
        """Cheap synchronous check so APIs can reject bad uploads before queueing."""
        return preprocess.load_image(data, self.config)

    def generate(self, data: bytes, on_progress: Optional[ProgressCallback] = None) -> GenerationResult:
        cfg = self.config
        timings: dict[str, float] = {}

        def report(stage: str, fraction: float = 0.0):
            if on_progress:
                lo, hi = STAGES[stage]
                on_progress(stage, lo + (hi - lo) * fraction)

        def timed(stage: str, fn):
            report(stage)
            t0 = time.perf_counter()
            result = fn()
            timings[stage] = round(time.perf_counter() - t0, 2)
            return result

        image = timed("validating_input", lambda: preprocess.load_image(data, cfg))
        del data  # do not keep the upload around longer than needed
        rgba = timed("removing_background", lambda: preprocess.isolate_subject(image, self.background_remover))
        framed = preprocess.frame_subject(rgba, cfg)
        del image, rgba

        recon = timed(
            "reconstructing",
            lambda: self.reconstructor.reconstruct(framed, lambda f: report("reconstructing", f)),
        )
        mesh = timed("processing_mesh", lambda: mesh_stage.clean_mesh(recon.mesh, cfg))
        placement = mesh_stage.placement_transform(mesh, cfg)
        textured = timed("texturing", lambda: texture.bake_texture(mesh, recon.color_fn, cfg))
        glb = timed("exporting", lambda: export.export_glb(textured, placement, cfg))

        glb_report = validate.validate_glb(glb, cfg.max_glb_bytes)
        if not glb_report.ok:
            raise ExportValidationError("Generated GLB failed validation: " + "; ".join(glb_report.errors))

        warnings = [
            "Likeness is not yet evaluated: TripoSR reconstructs overall shape and colour from a single view; "
            "fine facial detail and the unseen back side are approximations."
        ]
        return GenerationResult(
            glb=glb,
            reconstructor=self.reconstructor.name,
            stats=glb_report.stats,
            timings=timings,
            warnings=warnings,
        )
