import pytest

from charengine import CharacterPipeline, PipelineConfig
from helpers import SphereReconstructor, cutout


@pytest.fixture
def pipeline():
    return CharacterPipeline(
        PipelineConfig(texture_size=256, target_faces=2000),
        reconstructor=SphereReconstructor(),
        background_remover=cutout,
    )
