# Reproducing the measured runs

Run commands from the repository root after installation. Use fresh output
directories for reruns; the pipeline and evaluation scripts reject overwrites.

```powershell
python scripts/download_assets.py all
python scripts/download_dataset.py --revision e8b8cea002692efd74c945fcdad63e729adc5671
python scripts/train_detector.py --epochs 3 --device cpu --name soccer_yolo26n_cpu3
python scripts/evaluate_detector.py --output outputs/evaluation_reference
python scripts/evaluate_detector.py --model models/checkpoints/soccer_yolo26n_cpu3.pt --output outputs/evaluation_cpu3
python scripts/experiment_confidence.py
python scripts/run_video.py --input samples/input/match.mp4 --output outputs/videos/complete_match.mp4 --artifacts outputs/complete_demo
python scripts/benchmark.py --input samples/input/match.mp4 --models models/checkpoints/best.pt models/checkpoints/soccer_yolo26n_cpu3.pt --frames 30
python scripts/create_report.py --artifacts outputs/complete_demo --video outputs/videos/complete_match.mp4 --preview
```

The first training run used a relative Ultralytics project path, so its historical
files are under `runs/detect/outputs/training/soccer_yolo26n_cpu3/`. The training
script now resolves the project path explicitly; reruns write directly under
`outputs/training/soccer_yolo26n_cpu3/`. Both refer to the same run name, not two
different experiments. Run metadata and exported summaries preserve the actual
configuration. Numerical reproducibility can vary across library versions and
hardware; hashes and versions identify the assets used here.

To publish a fresh set of summary files (no raw dataset or video):

```powershell
python scripts/publish_results.py --training outputs/training/soccer_yolo26n_cpu3 --output reports/my_run
```

For CPU/GPU comparison, use `--devices cpu cuda:0` in the benchmark command after
installing a CUDA-enabled PyTorch build on a supported machine. No GPU is present
in the development environment, so the published comparison is CPU-only. Run
benchmarks without concurrent training/inference jobs; warm-up is excluded and
decoding is outside the inference timer.

For tracking evaluation, annotate every object in every frame of a chosen range
using CSV columns `frame,track_id,class,x1,y1,x2,y2`. Frames use zero-based indices
and boxes use original-image pixels. Empty frames have no rows but must still be
included in the explicit evaluated interval:

```powershell
python -m pip install -e ".[evaluation]"
python scripts/evaluate_tracking.py --truth data/tracking_truth.csv --predictions outputs/complete_demo/tracks.csv --start-frame 0 --end-frame 249
```

Missing annotations inside that interval are interpreted as empty ground truth,
so partially labeled frames are unsuitable for a meaningful score.
