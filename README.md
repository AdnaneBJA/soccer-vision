# SoccerVision

AI-Powered Soccer Detection, Tracking & Analytics — under development.

## Current status

Milestone 1 implements streaming video input/output, metadata inspection, a CLI,
logging, and synthetic MP4 integration tests. Detection, tracking, training and
analytics are future milestones; no model accuracy or soccer footage results
have been measured yet. The original `main.py` starter is preserved.

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

## Architecture

Current: input MP4 → VideoReader → BGR frames → VideoWriter → output MP4.

Planned: video → PyTorch detector → ByteTrack → teams → analytics → annotations.

## Project structure

- `src/soccer_vision/video/`: reader and writer with context-manager cleanup
- `src/soccer_vision/cli.py`: streaming pipeline and logging
- `scripts/run_video.py`: source-checkout entry point
- `tests/`: deterministic synthetic video checks
- `data/`, `models/`: instructions for future local assets
- `Agent.md`: project roadmap

## Next milestones

Integrate pretrained detection and bounding boxes, then select and document a
licensed soccer dataset for fine-tuning and evaluation. Add ByteTrack IDs and
CSV exports before team classification, possession estimates and heatmaps.
Report only actual evaluation and benchmark runs. Image-space motion will be
reported in pixels; physical speed/distance requires pitch calibration.
