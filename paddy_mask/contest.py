"""Contest CSV protocol and background-only integration APIs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from typing import Iterable, Sequence

import numpy as np

from .backends import (
    NoPaddyDetected,
)
from .factory import build_extractor


@dataclass(frozen=True)
class ContestInput:
    """An image entry declared by the contest's input.csv."""

    width: int
    height: int
    relative_image_path: str


@dataclass(frozen=True)
class ContestOutput:
    """Downstream-owned result values for one final contest output row."""

    input: ContestInput
    relative_output_path: str
    level: str
    p: str
    w: str
    r: str


@dataclass(frozen=True)
class ForegroundROI:
    """Primary paddy pixels retained for a downstream rice/weed classifier."""

    mask: np.ndarray
    masked_rgb: np.ndarray
    bounding_box: tuple[int, int, int, int]
    crop_rgb: np.ndarray
    crop_mask: np.ndarray


def _normalise_relative_path(relative_path: str) -> PurePosixPath:
    if not relative_path or not relative_path.strip():
        raise ValueError("Image paths in input.csv cannot be empty.")
    windows_path = PureWindowsPath(relative_path)
    posix_path = PurePosixPath(relative_path.replace("\\", "/"))
    if (
        windows_path.is_absolute()
        or windows_path.drive
        or posix_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ValueError(f"Image path must be a relative path inside the contest directory: {relative_path}")
    return posix_path


def input_image_path(working_directory: Path, relative_path: str) -> Path:
    """Resolve either slash convention from input.csv below the working directory."""

    return working_directory.joinpath(*_normalise_relative_path(relative_path).parts)


def output_relative_path(relative_image_path: str) -> str:
    """Return the required sibling `<stem>-output.JPG` path, preserving slash style."""

    normalised = _normalise_relative_path(relative_image_path)
    output_name = f"{normalised.stem}-output.JPG"
    result = str(normalised.with_name(output_name))
    return result.replace("/", "\\") if "\\" in relative_image_path else result


def final_output_path(working_directory: Path, relative_image_path: str) -> Path:
    """Resolve the required final-output JPEG location in the contest directory."""

    return input_image_path(working_directory, output_relative_path(relative_image_path))


def read_input_csv(path: Path) -> list[ContestInput]:
    """Read and validate the contest input.csv format exactly."""

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = [row for row in csv.reader(csv_file) if any(cell.strip() for cell in row)]
    if not rows or len(rows[0]) != 1:
        raise ValueError("input.csv must begin with one image count (N).")
    try:
        expected_count = int(rows[0][0])
    except ValueError as error:
        raise ValueError("The first input.csv row must contain an integer image count.") from error
    if expected_count < 0:
        raise ValueError("The input.csv image count cannot be negative.")
    if len(rows) - 1 != expected_count:
        raise ValueError(
            f"input.csv declares {expected_count} image(s), but contains {len(rows) - 1} image row(s)."
        )

    records: list[ContestInput] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != 3:
            raise ValueError(f"input.csv row {row_number} must contain width,height,relative image path.")
        try:
            width, height = int(row[0]), int(row[1])
        except ValueError as error:
            raise ValueError(f"input.csv row {row_number} has a non-integer width or height.") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"input.csv row {row_number} width and height must be positive.")
        relative_path = row[2].strip()
        _normalise_relative_path(relative_path)
        records.append(ContestInput(width, height, relative_path))
    return records


def write_output_csv(path: Path, results: Sequence[ContestOutput]) -> None:
    """Write final metadata supplied by the downstream rice/weed classifier."""

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([len(results)])
        for result in results:
            writer.writerow(
                [
                    result.input.width,
                    result.input.height,
                    result.relative_output_path,
                    result.level,
                    result.p,
                    result.w,
                    result.r,
                ]
            )


def build_foreground_roi(rgb_image: np.ndarray, foreground_mask: np.ndarray) -> ForegroundROI:
    """Blacken non-paddy pixels and return full-frame plus cropped foreground data."""

    rgb = np.asarray(rgb_image)
    mask = np.asarray(foreground_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3).")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("foreground_mask dimensions must match rgb_image.")
    if not mask.any():
        raise ValueError("foreground_mask cannot be empty.")

    y_coords, x_coords = np.nonzero(mask)
    left, top = int(x_coords.min()), int(y_coords.min())
    right, bottom = int(x_coords.max()) + 1, int(y_coords.max()) + 1
    masked_rgb = np.zeros_like(rgb)
    masked_rgb[mask] = rgb[mask]
    return ForegroundROI(
        mask=mask,
        masked_rgb=masked_rgb,
        bounding_box=(left, top, right, bottom),
        crop_rgb=masked_rgb[top:bottom, left:right].copy(),
        crop_mask=mask[top:bottom, left:right].copy(),
    )


def compose_contest_image(
    foreground: ForegroundROI,
    rice_mask: np.ndarray,
    weed_mask: np.ndarray,
) -> np.ndarray:
    """Compose the prescribed final RGB image from downstream classification masks."""

    rice = np.asarray(rice_mask, dtype=bool)
    weed = np.asarray(weed_mask, dtype=bool)
    if rice.shape != foreground.mask.shape or weed.shape != foreground.mask.shape:
        raise ValueError("rice_mask and weed_mask must match the full-frame foreground mask.")
    if np.any(rice & weed):
        raise ValueError("rice_mask and weed_mask must not overlap.")
    if np.any((rice | weed) & ~foreground.mask):
        raise ValueError("Downstream masks must not classify pixels outside the foreground ROI.")

    output = np.zeros((*foreground.mask.shape, 3), dtype=np.uint8)
    output[rice] = (128, 128, 128)
    output[weed] = (255, 255, 255)
    return output


def _load_rgb_image(path: Path) -> tuple[object, np.ndarray]:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError("Pillow is required. Install requirements-paddy-mask.txt.") from error
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    return image, np.asarray(image)


def _save_foreground_artifacts(roi: ForegroundROI, base_path: Path) -> None:
    from PIL import Image

    base_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(roi.mask, 255, 0).astype(np.uint8), mode="L").save(
        base_path.with_suffix(".foreground-mask.png")
    )
    Image.fromarray(roi.masked_rgb, mode="RGB").save(base_path.with_suffix(".foreground.png"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create background-masked foreground artifacts from contest input.csv."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("input.csv"))
    parser.add_argument(
        "--foreground-dir",
        type=Path,
        default=Path(".paddy-mask-foreground"),
        help="Directory for foreground PNG artifacts; never creates final contest outputs.",
    )
    parser.add_argument("--backend", choices=("auto", "zero-shot", "heuristic"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        csv_path = args.input_csv.resolve()
        working_directory = csv_path.parent
        records = read_input_csv(csv_path)
        extractor, backend = build_extractor(
            args.backend,
            args.device,
            args.grounding_model_id,
            args.sam_model_id,
            args.box_threshold,
            args.text_threshold,
        )
        for record in records:
            image_path = input_image_path(working_directory, record.relative_image_path)
            image, rgb = _load_rgb_image(image_path)
            if image.size != (record.width, record.height):
                raise ValueError(
                    f"{record.relative_image_path} is {image.size[0]}x{image.size[1]}, "
                    f"but input.csv declares {record.width}x{record.height}."
                )
            result = extractor.extract(image)
            roi = build_foreground_roi(rgb, result.mask)
            artifact_path = input_image_path(args.foreground_dir, record.relative_image_path)
            _save_foreground_artifacts(roi, artifact_path)
            print(f"{record.relative_image_path} -> {artifact_path} [{result.backend}]")
        print(
            f"Prepared {len(records)} foreground ROI artifact set(s) using {backend}. "
            "Final -output.JPG files and output.csv remain downstream-owned."
        )
        return 0
    except (NoPaddyDetected, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
