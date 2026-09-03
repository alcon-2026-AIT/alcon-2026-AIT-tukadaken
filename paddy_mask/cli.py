"""Command-line entry point for the automatic paddy-mask extractor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from .backends import (
    NoPaddyDetected,
)
from .factory import build_extractor


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one primary rice-paddy mask from each RGB photograph."
    )
    parser.add_argument("--input", required=True, type=Path, help="An image file or a directory of images.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for binary PNG masks.")
    parser.add_argument(
        "--backend",
        choices=("auto", "zero-shot", "heuristic"),
        default="auto",
        help="auto uses zero-shot models, falling back only when their packages are absent.",
    )
    parser.add_argument("--recursive", action="store_true", help="Traverse input directories recursively.")
    parser.add_argument("--device", default="auto", help="Torch device for zero-shot mode: auto, cpu, mps, or cuda.")
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    return parser


def _input_images(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {source}")
        return [source]
    if not source.is_dir():
        raise ValueError(f"Input path does not exist: {source}")
    paths = source.rglob("*") if recursive else source.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)


def _output_path(image_path: Path, source: Path, output: Path) -> Path:
    relative = image_path.name if source.is_file() else image_path.relative_to(source)
    return output / Path(relative).with_suffix(".mask.png")


def _load_image(path: Path) -> object:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError("Pillow is required. Install requirements-paddy-mask.txt.") from error
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _save_mask(mask: np.ndarray, destination: Path) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(destination)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.input.is_dir() and (
            args.output.resolve() == args.input.resolve()
            or args.output.resolve().is_relative_to(args.input.resolve())
        ):
            raise ValueError("Output directory must be outside the input directory.")
        image_paths = list(_input_images(args.input, args.recursive))
        if not image_paths:
            raise ValueError(f"No supported images found in: {args.input}")
        extractor, mode = build_extractor(
            args.backend,
            args.device,
            args.grounding_model_id,
            args.sam_model_id,
            args.box_threshold,
            args.text_threshold,
        )
        for image_path in image_paths:
            result = extractor.extract(_load_image(image_path))
            destination = _output_path(image_path, args.input, args.output)
            _save_mask(result.mask, destination)
            print(
                f"{image_path} -> {destination} "
                f"[{result.backend}; score={result.selected.score:.3f}]"
            )
        print(f"Created {len(image_paths)} binary mask(s) using {mode}.")
        return 0
    except (NoPaddyDetected, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
