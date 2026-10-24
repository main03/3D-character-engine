# CLAUDE.md

## Project: Custom Image-to-3D Character Engine

### Goal
Build a developer-friendly library that accepts a user's photo and generates a stylized 3D human character with recognizable facial features, exported as a `.glb` file.

The character should be usable in Three.js, React Three Fiber, Blender, and other GLB-compatible applications.

### Core Requirements
1. Accept PNG and JPG images.
2. Generate a 3D human character from an uploaded image.
3. Preserve the user's visual likeness as closely as possible, including face shape, skin tone, hairstyle, eyes, and nose.
4. Generate textured 3D geometry with suitable materials.
5. Export a valid GLB file containing the mesh, materials, and textures.
6. Provide a simple 3D preview with rotation and zoom.
7. Keep the architecture modular so models and processing stages can be replaced independently.

### Technology Stack
- Frontend: Next.js, TypeScript, React Three Fiber, Three.js
- Backend: Python, FastAPI
- AI inference: PyTorch and a suitable pretrained image-to-3D model
- 3D processing: Blender Python API and appropriate mesh-processing libraries
- Output: GLB

### Development Principles
- Research and evaluate existing open-source models before implementing custom AI.
- Never pretend a placeholder or generic character is a successful personalized reconstruction.
- Keep AI inference separate from the API and frontend.
- Validate uploaded files and generated GLB assets.
- Handle errors, loading states, and generation progress.
- Check model licenses and GPU requirements before integration.
- Protect user-uploaded images and avoid retaining them unnecessarily.
- Do not introduce unnecessary dependencies or infrastructure.

### Implementation Plan
**Phase 1:** Inspect the repository and establish the project structure.

**Phase 2:** Build an end-to-end prototype: upload an image, run a pretrained image-to-3D model, export a GLB, and preview it in the browser.

**Phase 3:** Evaluate facial likeness and improve face reconstruction separately where needed.

**Phase 4:** Add model optimization, robust GLB validation, asynchronous generation, and a developer SDK.

### Working notes
- Layout: `engine/` (charengine: AI + 3D processing, no web code), `server/` (FastAPI transport only), `web/` (Next.js).
- Python: `uv sync`, tests `uv run pytest` (fast; use the `SphereReconstructor` test double, never as a runtime fallback).
- Web: `cd web && npm run dev`; check with `npx tsc --noEmit && npx eslint src`. Next.js 16: read `web/node_modules/next/dist/docs/` before using unfamiliar APIs.
- Dev machine has no CUDA GPU and no C++ compiler: torch is CPU-only, TripoSR is vendored in
  `engine/charengine/vendor/tsr` with PyMCubes instead of torchmcubes (see its NOTICE.md).
- Canonical mesh frame: Y-up, character faces +Z; placed standing on y=0 with height `target_height`.
- TripoSR input camera in the canonical frame: (0, 0, 1.8), FOV 40°, fitted by silhouette IoU; photo projection
  (`stages/projection.py`) depends on it. Re-fit it if the reconstructor or framing changes.
- In R3F, don't put drei `<Html>` in a Suspense fallback (React 19 unmount race); use a DOM overlay.
- Background removal uses `isnet-general-use` (Apache-2.0); rembg's default `bria-rmbg` is non-commercial.
