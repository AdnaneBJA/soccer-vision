# SoccerVision

AI-Powered Soccer Detection, Tracking & Analytics — under development.

## Current status

Milestones 1–2 implement streaming video input/output, metadata inspection, a CLI,
logging, pretrained PyTorch detection, and annotated MP4 output. Tracking,
soccer-specific training and analytics are future milestones; no soccer accuracy
results have been measured yet. The original `main.py` starter is preserved.

## Installation

Python 3.11+ is recommended; Python 3.10 is also supported for the existing
development environment. Core dependencies are OpenCV and NumPy.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If an existing environment has no pip, run `python -m ensurepip --upgrade` first.
Select this environment as the project interpreter in PyCharm.

## Usage

Place a legally obtained local MP4 in `samples/input/`, then run:

```powershell
python scripts/run_video.py --input samples/input/match.mp4 --output outputs/videos/copy.mp4
```

After installation, `soccer-vision --input ... --output ...` is also available.

For detection, install the optional dependencies and select detection mode:

```powershell
python -m pip install -e ".[detection]"
python scripts/run_video.py --mode detect --input samples/input/match.mp4 --output outputs/videos/detected.mp4 --device auto --confidence 0.35
```

The default `yolo26n.pt` downloads on first use and is ignored by Git. It is a
[pretrained YOLO26 nano detector](https://docs.ultralytics.com/models/yolo26/).
Generic COCO `person` labels remain `person`; `sports ball` becomes `ball`.
Referees and goalkeepers cannot be distinguished by these pretrained labels.
Use `--model models/checkpoints/best.pt` for a local soccer detection checkpoint
with player/goalkeeper/referee/ball labels. Other unrelated classes are filtered.
The adapter uses the documented [prediction API](https://docs.ultralytics.com/modes/predict/)
and returns bounding boxes in original-image pixels. `--image-size` defaults to
640 and must be a positive multiple of 32. CPU works without CUDA; `auto` selects
CUDA when PyTorch reports it available. Explicit unavailable devices fail clearly.
A CUDA-enabled PyTorch installation is required for GPU inference.

`--mode copy` (the default) retains the original video-only workflow.
Output parent directories are created automatically. Use a new output filename
for each run. Frames are re-encoded using MP4V, preserving nominal FPS and frame
dimensions. Re-encoding is lossy; the file is not a byte-identical copy.
Audio is not copied. Constant-frame-rate video and even dimensions are required;
variable-frame-rate timestamps are not preserved. OpenCV can stop decoding at a
corrupt frame; the CLI reports an error when decoded and reported counts differ.
An incomplete output may remain after an error.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests generate a local 20-frame MP4, copy it through the pipeline, and verify
frame count, dimensions, FPS, duration and approximate image content. They also
check input errors, frame validation, resource closure and CLI exit codes.
These are video plumbing tests, not soccer detection evaluation.

Detection tests also verify confidence/class filtering, coordinate conversion,
device selection, and annotation behavior without downloading model weights.
Run a real model smoke test with `python scripts/smoke_detection.py`; it uses
Ultralytics' bundled bus image and writes a unique folder under `outputs/smoke/`.
The verified Windows CPU run used Ultralytics 8.4.141, PyTorch 2.14.0+cpu and
OpenCV 4.14.0: four image detections and a three-frame annotated MP4 at 25 FPS.
This is an integration check, not a soccer accuracy or performance benchmark.
CUDA hardware execution has not been tested.

## Architecture

Current: input MP4 → VideoReader → optional detector/annotations → VideoWriter → output MP4.

Planned: video → PyTorch detector → ByteTrack → teams → analytics → annotations.

## Project structure

- `src/soccer_vision/video/`: reader and writer with context-manager cleanup
- `src/soccer_vision/cli.py`: streaming pipeline and logging
- `src/soccer_vision/detection/`: device selection and PyTorch model adapter
- `src/soccer_vision/visualization/`: boxes, labels and ball markers
- `scripts/run_video.py`: source-checkout entry point
- `tests/`: deterministic synthetic video checks
- `data/`, `models/`: instructions for future local assets
- `Agent.md`: project roadmap

## Next milestones

Select and document a licensed soccer dataset for fine-tuning and evaluation.
Add ByteTrack IDs and
CSV exports before team classification, possession estimates and heatmaps.
Report only actual evaluation and benchmark runs. Image-space motion will be
reported in pixels; physical speed/distance requires pitch calibration.
