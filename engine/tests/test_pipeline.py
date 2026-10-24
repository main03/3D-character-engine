import io

import numpy as np
import pytest
import trimesh

from charengine import InputImageError
from charengine.stages.validate import validate_glb

from helpers import png_bytes


def test_generate_produces_valid_textured_glb(pipeline):
    stages = []
    result = pipeline.generate(png_bytes(), lambda s, p: stages.append((s, p)))

    report = validate_glb(result.glb, 10**8)
    assert report.ok, report.errors
    assert [s for s, _ in stages][0] == "validating_input"
    assert all(a <= b for (_, a), (_, b) in zip(stages, stages[1:]))

    mesh = trimesh.load(io.BytesIO(result.glb), file_type="glb", force="mesh")
    assert mesh.bounds[0][1] == pytest.approx(0.0, abs=1e-6)
    assert mesh.extents[1] == pytest.approx(1.0, rel=1e-3)
    # The baked texture should reproduce the position-based colours, not be blank.
    tex = np.asarray(mesh.visual.material.baseColorTexture)
    assert tex[..., :3].std() > 20


@pytest.mark.parametrize(
    "data",
    [b"", b"not an image", png_bytes(size=(64, 64)), png_bytes(fmt="GIF", mode="RGB")],
    ids=["empty", "garbage", "too-small", "gif"],
)
def test_rejects_bad_input(pipeline, data):
    with pytest.raises(InputImageError):
        pipeline.validate_input(data)


def test_rejects_image_without_subject(pipeline):
    transparent = png_bytes(mode="RGBA")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(buf, format="PNG")
    with pytest.raises(InputImageError):
        pipeline.generate(buf.getvalue())
    assert transparent  # opaque RGBA still accepted by validation
    pipeline.validate_input(transparent)


def test_rejects_corrupt_glb():
    assert not validate_glb(b"glTF" + b"\0" * 20, 10**6).ok
    assert not validate_glb(b"nope", 10**6).ok
