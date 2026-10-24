# 3D Character Engine

Upload a photo of a person and get a textured 3D character as a `.glb`, previewed in the browser.

```
photo ─▶ validate ─▶ remove background ─▶ TripoSR (image→3D) ─▶ smooth, clean & decimate ─▶ UV unwrap ─▶ bake texture (photo projected on visible front + model colours elsewhere) ─▶ GLB ─▶ validate
```

## Status

Phase 2 prototype, plus a first Phase 3 step. Shape comes from TripoSR. The camera-facing surface is textured by
projecting the original photo through TripoSR's input camera (fitted at distance 1.8, FOV 40°), so eyes, brows and
skin come from the photo itself instead of the model's blurry 512 px prediction. **Face geometry is still coarse**
and the sides and back are the model's guess. Likeness has not been measured yet.

## Layout

| Path | What |
| --- | --- |
| `engine/charengine/` | Python library: pipeline stages, swappable `Reconstructor` backends, GLB validation, CLI |
| `engine/charengine/vendor/tsr/` | Vendored TripoSR model code (MIT), see `NOTICE.md` |
| `server/app/` | FastAPI async job API (no AI code) |
| `web/` | Next.js + React Three Fiber UI |

## Run it

Requirements: Python 3.11–3.12 with [uv](https://docs.astral.sh/uv/), Node 20+. No GPU required (uses CUDA automatically if present).

```bash
uv sync                                                          # Python deps (CPU PyTorch)
uv run uvicorn app.main:app --app-dir server --port 8000          # API
cd web && npm install && npm run dev                              # UI on http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` if the API is not on `http://localhost:8000`, and `CHARENGINE_CORS_ORIGINS`
(comma-separated) if the UI is not on `http://localhost:3000`.

The first generation downloads model weights (~1.7 GB TripoSR from Hugging Face, ~170 MB background-removal model).
On a laptop CPU a generation takes roughly 1–2 minutes.

CLI without the server:

```bash
uv run charengine generate photo.jpg -o character.glb
```

## API

| Method | Path | |
| --- | --- | --- |
| `POST` | `/api/v1/jobs` | multipart field `image` (PNG/JPG ≤ 15 MB) → `202 {id, status}` |
| `GET` | `/api/v1/jobs/{id}` | `status` (`queued`/`running`/`succeeded`/`failed`), `stage`, `progress`, `error`, `result` |
| `GET` | `/api/v1/jobs/{id}/model.glb` | the generated model |

Uploads are held in memory only and discarded once processed; results are kept in memory for one hour.

## Output

A single GLB: one mesh (~40k triangles) with normals and UVs, one PBR material with a 2048² base-colour
texture. Y-up, facing +Z, feet/base at y = 0, height 1.0. Every file is validated (structure, attributes,
embedded texture, finite bounds) before it is returned.

## Models and licences

- **TripoSR** (Stability AI & Tripo AI): MIT code and weights. Chosen because it runs on CPU; heavier
  models (TRELLIS, Hunyuan3D-2, SF3D) need a CUDA GPU and some have restrictive licences.
- **rembg** with `isnet-general-use`: MIT / Apache-2.0.

## Tests

```bash
uv run pytest           # engine + API, uses a test double instead of the real model (seconds)
cd web && npx tsc --noEmit && npx eslint src
```
