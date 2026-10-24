"""Input validation and subject isolation."""

import io
import threading

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import PipelineConfig
from ..errors import InputImageError

ALLOWED_FORMATS = {"PNG", "JPEG"}

Image.MAX_IMAGE_PIXELS = 8192 * 8192


def load_image(data: bytes, config: PipelineConfig) -> Image.Image:
    """Decode and validate an uploaded PNG/JPG. Returns an RGB or RGBA image."""
    if not data:
        raise InputImageError("The uploaded file is empty.")
    if len(data) > config.max_upload_bytes:
        raise InputImageError(
            f"Image is larger than {config.max_upload_bytes // (1024 * 1024)} MB."
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InputImageError("File is not a readable image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise InputImageError(f"Unsupported image format {fmt!r}; use PNG or JPG.")

    image = ImageOps.exif_transpose(image)
    w, h = image.size
    if min(w, h) < config.min_image_side:
        raise InputImageError(f"Image is too small ({w}x{h}); minimum side is {config.min_image_side}px.")
    if max(w, h) > config.max_image_side:
        raise InputImageError(f"Image is too large ({w}x{h}); maximum side is {config.max_image_side}px.")

    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    return image.convert("RGBA" if has_alpha else "RGB")


class BackgroundRemover:
    """Lazily-loaded rembg session (the ONNX model is downloaded on first use)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._session = None
        self._lock = threading.Lock()

    def __call__(self, image: Image.Image) -> Image.Image:
        import rembg

        with self._lock:
            if self._session is None:
                self._session = rembg.new_session(self.model_name)
            return rembg.remove(image, session=self._session)


def isolate_subject(image: Image.Image, remover: BackgroundRemover) -> Image.Image:
    """Return an RGBA image with the background removed (kept as-is if already transparent)."""
    if image.mode == "RGBA" and np.asarray(image)[..., 3].min() < 255:
        return image
    return remover(image.convert("RGB")).convert("RGBA")


def frame_subject(rgba: Image.Image, config: PipelineConfig) -> Image.Image:
    """Crop to the subject, pad to a square with the configured margin, and composite on grey.

    TripoSR expects the object centred on a neutral (0.5 grey) background.
    """
    arr = np.asarray(rgba)
    alpha = arr[..., 3]
    ys, xs = np.nonzero(alpha > 127)
    if len(ys) < 0.01 * alpha.size:
        raise InputImageError("Could not find a person in the image. Use a photo with a clearly visible subject.")
    fg = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    side = int(max(fg.shape[:2]) / config.foreground_ratio)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    y0, x0 = (side - fg.shape[0]) // 2, (side - fg.shape[1]) // 2
    canvas[y0 : y0 + fg.shape[0], x0 : x0 + fg.shape[1]] = fg

    rgb = canvas[..., :3].astype(np.float32) / 255.0
    a = canvas[..., 3:4].astype(np.float32) / 255.0
    composited = rgb * a + 0.5 * (1.0 - a)
    out = Image.fromarray((composited * 255.0).round().astype(np.uint8))
    return out.resize((config.model_input_size, config.model_input_size), Image.LANCZOS)
