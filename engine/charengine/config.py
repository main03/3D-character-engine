from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    # Input limits
    max_upload_bytes: int = 15 * 1024 * 1024
    min_image_side: int = 128
    max_image_side: int = 8192

    # Preprocessing
    # isnet-general-use is Apache-2.0. Avoid rembg's default (bria-rmbg), which is non-commercial.
    background_model: str = "isnet-general-use"
    foreground_ratio: float = 0.85
    model_input_size: int = 512

    # Reconstruction (TripoSR)
    triposr_repo: str = "stabilityai/TripoSR"
    mc_resolution: int = 256
    mc_threshold: float = 25.0
    chunk_size: int = 8192

    # Mesh post-processing
    target_faces: int = 40_000
    min_component_ratio: float = 0.02
    target_height: float = 1.0

    # Texturing
    texture_size: int = 1024
    roughness: float = 0.8

    # Output limits
    max_glb_bytes: int = 50 * 1024 * 1024
