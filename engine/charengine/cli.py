import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .errors import CharEngineError
from .pipeline import CharacterPipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="charengine", description="Generate a 3D character (.glb) from a photo.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("image", type=Path)
    gen.add_argument("-o", "--output", type=Path, default=None)
    gen.add_argument("--mc-resolution", type=int, default=PipelineConfig.mc_resolution)
    gen.add_argument("--texture-size", type=int, default=PipelineConfig.texture_size)
    args = parser.parse_args(argv)

    config = PipelineConfig(mc_resolution=args.mc_resolution, texture_size=args.texture_size)
    pipeline = CharacterPipeline(config)
    output = args.output or args.image.with_suffix(".glb")
    try:
        result = pipeline.generate(args.image.read_bytes(), lambda s, p: print(f"[{p:5.0%}] {s}", file=sys.stderr))
    except CharEngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output.write_bytes(result.glb)
    print(f"wrote {output} {result.stats} timings={result.timings}")
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
