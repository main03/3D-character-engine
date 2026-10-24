import time

from fastapi.testclient import TestClient

from app.main import create_app
from charengine import CharacterPipeline, PipelineConfig
from charengine.stages.validate import validate_glb
from helpers import SphereReconstructor, cutout, png_bytes


def make_client():
    pipeline = CharacterPipeline(PipelineConfig(texture_size=256, target_faces=2000), SphereReconstructor(), cutout)
    return TestClient(create_app(pipeline))


def test_job_lifecycle():
    with make_client() as client:
        r = client.post("/api/v1/jobs", files={"image": ("me.png", png_bytes(), "image/png")})
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]
        for _ in range(100):
            body = client.get(f"/api/v1/jobs/{job_id}").json()
            if body["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert body["status"] == "succeeded", body
        glb = client.get(body["result"]["glb_url"])
        assert glb.headers["content-type"] == "model/gltf-binary"
        assert validate_glb(glb.content, 10**8).ok


def test_rejects_bad_uploads():
    with make_client() as client:
        assert client.post("/api/v1/jobs", files={"image": ("x.gif", b"GIF89a", "image/gif")}).status_code == 415
        assert client.post("/api/v1/jobs", files={"image": ("x.png", b"junk", "image/png")}).status_code == 400
        assert client.get("/api/v1/jobs/missing").status_code == 404
