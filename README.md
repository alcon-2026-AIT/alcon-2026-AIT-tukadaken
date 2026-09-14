# alcon-2026-AIT-tukadaken

## Automatic primary rice-paddy masks

`paddy_mask` is a standalone, fully automatic command-line tool that writes
one binary mask for the main photographed rice paddy in every RGB image. It
does not require clicks, bounding boxes, or training labels, and never reads
any folder unless it is passed with `--input`.

The default semantic pipeline uses Grounding DINO to propose rice-paddy and
scene-object regions, then Segment Anything (SAM) to produce masks. It selects
one mask by detector confidence, lower-centre placement, useful area, contact
with the bottom of the image, and penalties for overlapping detected sky,
forest, roads, and buildings. Specific `paddy field`/`rice field` detections
are preferred over broader `agricultural field` proposals. This helps prevent a
visually similar forest or an adjacent paddy from becoming the output.

### Install

For zero-shot semantic extraction (recommended), use Python 3.10--3.13:

```bash
python -m pip install -r requirements-paddy-mask-zero-shot.txt
```

The first zero-shot run downloads `IDEA-Research/grounding-dino-tiny` and
`facebook/sam-vit-base` from Hugging Face. Model availability, viewpoint, and
occlusion still affect results; inspect masks before using them for measurement.

### Run

```bash
python -m paddy_mask \
  --input /path/to/paddy_photos \
  --output /path/to/paddy_masks \
  --recursive
```

Each output is an 8-bit single-channel PNG named `<input-name>.mask.png`
(`0` = background; `255` = selected paddy). To process one photo, pass its
file path to `--input`. `--device auto` chooses CUDA, Apple MPS, or CPU; use
`--device cpu` to force CPU.

## Contest integration: background stage only

The contest executable reads `input.csv` in its current working directory:

```text
N
width,height,relative-image-path
```

`paddy_mask.contest` supports both `/` and Windows `\` separators in those
relative paths. It validates `N`, dimensions, and paths, then extracts the
primary-paddy foreground ROI without inventing any rice/weed measurements:

```bash
python -m paddy_mask.contest --input-csv input.csv
```

This produces `.paddy-mask-foreground/<input-relative-stem>.foreground-mask.png`
and `.foreground.png`. The mask is `0`/`255`; the RGB artifact preserves only
the ROI and forces every other/background pixel to black. Use
`--foreground-dir /path/to/staging` to choose another staging directory.
The command intentionally does **not** create final `-output.JPG` images or
`output.csv`, because classification and numeric values belong to the
downstream rice/weed component.

### Downstream integration contract

The background module exposes the following APIs from `paddy_mask.contest`:

```python
roi = build_foreground_roi(original_rgb, primary_paddy_mask)
# Classifier returns full-frame, boolean, mutually-exclusive masks inside roi.mask.
final_rgb = compose_contest_image(roi, rice_mask, weed_mask)
```

`compose_contest_image` enforces the final color contract: rice is
`(128, 128, 128)`, weed is `(255, 255, 255)`, and every background or
unclassified pixel is `(0, 0, 0)`. It rejects downstream masks that overlap or
extend beyond the foreground ROI. Save `final_rgb` at
`final_output_path(cwd, relative_image_path)`, which yields the required
`<stem>-output.JPG` beside the source image.

The downstream component supplies `level`, `p`, `w`, and `r`; this module
never estimates them. Pass its real values through `ContestOutput` and
`write_output_csv(Path("output.csv"), results)` to write:

```text
N
width,height,relative-output-filename,level,p,w,r
```

## Unified contest executable

`paddy_mask.contest_runner` is the executable integration point for a supplied
two-class rice/weed model. It reads `input.csv` from the current directory by
default, writes each required sibling `<stem>-output.JPG`, and writes
`output.csv` in that same directory:

```bash
python -m pip install -r requirements-paddy-mask-contest.txt
python -m paddy_mask.contest_runner --model /path/to/rice_weed_model.pkl
```

The model must be a `joblib` model trained for the seven BGR/Lab/ExG features
from the supplied `detector.py` interface, with class `0` for rice and class
`1` for weed. The original supplied inference feature implementation is adapted
in `paddy_mask/rice_weed.py`; no license was supplied with that source, so
retain its provenance before redistributing it.

For each selected foreground paddy pixel, the model determines final color:
rice `(128,128,128)`, weed `(255,255,255)`, and all background pixels
`(0,0,0)`. `p` is the number of classified plant pixels, `w` is the number of
weed pixels, `r` is `w / p * 100`, and `level` follows the supplied thresholds
(`0:<8`, `1:<20`, `2:<40`, otherwise `3`). JPEG is required by the contest;
outputs are encoded at quality 100 without chroma subsampling.

No serialized rice/weed model was supplied to this repository. The unified
command deliberately fails without `--model` rather than fabricating rice or
weed labels. Use `python -m paddy_mask.contest` to prepare foreground-only
artifacts until a compatible model is available.

### Dependency fallback and failure behaviour

`--backend auto` is semantic zero-shot mode by default. It uses the
`heuristic-fallback` only when `torch` or `transformers` is not installed and
prints a warning that the result is **not semantic**. The fallback is a
green-region heuristic and can confuse forest and paddy.

Use `--backend zero-shot` to fail instead of falling back, or
`--backend heuristic` only when that limitation is acceptable. If installed
zero-shot models run but find no paddy candidate, the command fails rather than
silently producing a heuristic result.

Keep `--output` outside an input directory so generated masks are never treated
as source photographs on a later run.

### Tests

The scoring, contest CSV, and foreground composition logic have no model dependency:

```bash
python -m unittest discover -s tests
```