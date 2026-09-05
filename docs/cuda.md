# Verified CUDA setup

The development machine has an NVIDIA GeForce RTX 3070 Ti with 8 GB VRAM and
driver 591.86. The earlier CUDA-unavailable result came from CPU-only PyTorch;
it did not establish that the machine lacked a GPU.

The project environment now uses PyTorch **2.11.0+cu128**, torchvision
**0.26.0+cu128**, and CUDA runtime **12.8**. These matching Windows/Python 3.10
wheels were available from the official CUDA 12.8 index. The old pip metadata
resolver failed during the first attempt; upgrading pip resolved the issue.

From the repository root, with Python processes using this environment stopped:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip check
```

Verify the actual device and allocation:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.ones(3, device='cuda'))"
```

All checks passed, including allocation on `cuda:0`. The existing `device: auto`
configuration now selects CUDA automatically; no application code changes were
needed. All 22 tests also passed with the new PyTorch version.

## Completed GPU run

```powershell
.\.venv\Scripts\python.exe scripts/run_video.py --input samples/input/match.mp4 --output outputs/videos/cuda_match.mp4 --artifacts outputs/cuda_demo --device cuda:0
.\.venv\Scripts\python.exe scripts/verify_run.py --artifacts outputs/cuda_demo --video outputs/videos/cuda_match.mp4
```

The verified output has 750 frames at 25 FPS (30 seconds), 17,904 track records
and 124 distinct track IDs. Open `outputs/cuda_demo/report.html` for its report.
Use new filenames for another run; the program rejects existing outputs.

Video processing, including model initialization, decoding, tracking, annotations
and encoding, took 36.94 seconds (**20.31 pipeline FPS**). Including heatmap export
took 48.96 seconds. These are measured runtime values, not the output playback FPS.
The run's detailed measurements are in `reports/cuda/demo_benchmark.json`.

## Same-model CPU/GPU comparison

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --input samples/input/match.mp4 --devices cpu cuda:0 --frames 30 --output outputs/metrics/cpu_vs_cuda.json
```

Both devices used the reference soccer checkpoint, the same 30 initial frames,
640-pixel inference size and PyTorch 2.11.0+cu128, after three warm-up predictions.

| Device | Mean latency | p95 latency | Inference FPS |
|---|---:|---:|---:|
| CPU | 229.96 ms | 257.03 ms | 4.35 |
| RTX 3070 Ti | 18.81 ms | 20.37 ms | 53.15 |

GPU inference throughput was approximately **12.2×** CPU throughput in this run.
The timer includes preprocessing, inference, postprocessing and transfer of
results to CPU; decoding and rendering are excluded. GPU memory is PyTorch's
peak allocated memory (330.63 MiB), not total device memory usage. Raw results
are preserved in `reports/cuda/cpu_vs_cuda.json`.

No new training or detection-accuracy evaluation was performed during this CUDA
setup. The existing model weights were reused. For GPU fine-tuning, use
`python scripts/train_detector.py --device cuda:0 --name soccer_yolo26n_cuda`.
