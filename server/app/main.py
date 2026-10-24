"""HTTP API. The pipeline (AI inference) lives in charengine; this module only handles transport."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from charengine import CharacterPipeline, InputImageError, PipelineConfig

from .jobs import JobManager, QueueFull

logging.basicConfig(level=logging.INFO)

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


def create_app(pipeline: CharacterPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jobs = JobManager(pipeline or CharacterPipeline(PipelineConfig()))
        yield
        app.state.jobs.shutdown()

    app = FastAPI(title="Character Engine API", version="0.1.0", lifespan=lifespan)
    origins = [o for o in os.environ.get("CHARENGINE_CORS_ORIGINS", "http://localhost:3000").split(",") if o]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    def jobs(request: Request) -> JobManager:
        return request.app.state.jobs

    def job_payload(request: Request, job) -> dict:
        body = {
            "id": job.id,
            "status": job.status,
            "stage": job.stage,
            "progress": round(job.progress, 3),
            "error": job.error,
        }
        if job.status == "queued":
            body["queue_position"] = jobs(request).queue_position(job)
        if job.status == "succeeded":
            body["result"] = {
                "glb_url": f"/api/v1/jobs/{job.id}/model.glb",
                "reconstructor": job.reconstructor,
                "stats": job.stats,
                "timings": job.timings,
                "warnings": job.warnings,
            }
        return body

    @app.get("/api/v1/health")
    def health(request: Request):
        return {"status": "ok"}

    @app.post("/api/v1/jobs", status_code=202)
    async def create_job(request: Request, image: UploadFile):
        manager = jobs(request)
        limit = manager.pipeline.config.max_upload_bytes
        if image.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "Upload a PNG or JPG image.")
        data = await image.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"Image is larger than {limit // (1024 * 1024)} MB.")
        try:
            manager.pipeline.validate_input(data)
        except InputImageError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            job = manager.submit(data)
        except QueueFull:
            raise HTTPException(429, "The generator is busy. Try again in a few minutes.")
        return job_payload(request, job)

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(request: Request, job_id: str):
        job = jobs(request).get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found (results expire after an hour).")
        return job_payload(request, job)

    @app.get("/api/v1/jobs/{job_id}/model.glb")
    def get_model(request: Request, job_id: str):
        job = jobs(request).get(job_id)
        if job is None or job.glb is None:
            raise HTTPException(404, "Model not available.")
        return Response(
            job.glb,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": f'attachment; filename="character-{job.id[:8]}.glb"', "Cache-Control": "no-store"},
        )

    return app


app = create_app()
