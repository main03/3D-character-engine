# Vendored TripoSR

Source: https://github.com/VAST-AI-Research/TripoSR @ 107cefdc244c39106fa830359024f6a2f1c78871
License: MIT (see LICENSE). Weights: https://huggingface.co/stabilityai/TripoSR (MIT).

Local modifications (search for `charengine:` comments):
- `models/isosurface.py`: marching cubes via PyMCubes instead of torchmcubes.
- `utils.py`: `find_class` resolves `tsr.*` config paths to this package; removed the
  rembg/imageio helpers (preprocessing lives in `charengine.stages.preprocess`).
- `bake_texture.py` removed (OpenGL-based); texture baking lives in `charengine.stages.texture`.
