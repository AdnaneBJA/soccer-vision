# Measured results

`latest/` contains summaries from actual local runs, not example numbers.
See `docs/reproduction.md` for commands, sources, revision and methodological limits.
Generated videos, source images and model weights stay outside Git.

## Detection and inference

| Model | Test mAP@50 | Test mAP@50:95 | Mean CPU latency | Inference FPS |
|---|---:|---:|---:|---:|
| Downloaded soccer YOLOv8x | 0.815 | 0.545 | 328.06 ms | 3.05 |
| Locally fine-tuned YOLO26n, 3 epochs | 0.183 | 0.095 | 20.77 ms | 48.15 |

Detection scores use the same 25-image test split. Benchmarks use the same first
30 decoded match frames at inference size 640, with three warm-up predictions.
Host: Windows, AMD Ryzen 9 7900X CPU, Python 3.10.20, PyTorch 2.14.0+cpu.
The smaller model is faster but insufficiently trained; these are not equivalent
accuracy operating points. Its ball and goalkeeper detection remain poor. The
reference checkpoint remains the default. This table preserves the original CPU
experiment. A later [CUDA benchmark](cuda/cpu_vs_cuda.json) measured the reference
model at 4.35 CPU FPS and 53.15 GPU FPS using PyTorch 2.11.0+cu128 on both devices.

Potential overlap between reference-model training footage and this dataset
prevents interpreting these as independent generalization scores.

## Complete demo

The 750-frame, 30-second clip was fully processed. The run emitted 124 distinct
track IDs; this includes non-player objects and fragmented tracks, not 124 real
players. The ball tracker emitted observed positions in 501/750 frames. This is
availability, not ground-truth ball recall. Possession was assigned on 302 frames
and remained unknown on 448, so the percentages do not summarize all match time.

Full-pipeline throughput was 3.75 FPS; steady-state prediction throughput in that
longer run was 4.22 FPS. A separate 30-frame benchmark produced 3.05 FPS for the
reference model. Timing varies with input frames, warm-up and system load; no
optimization claim is inferred from differences between these runs.

## Training and experiment

The local training run completed all three requested CPU epochs on 298 real
training images, validating on 49 images. `training_metrics.json` and
`training_curves.png` preserve actual losses and validation metrics. This short
run establishes the working training path, not a converged production model.

`confidence_experiment.json` contains all five threshold settings, evaluated on
49 validation images. Threshold 0.20 yields ball precision/recall of 0.594/0.422;
0.40 gives 0.720/0.400. Raising the threshold reduces false positives but does
not solve low ball recall. The experiment uses fixed-IoU matching, not mAP.

Real tracking MOTA/IDF1, team-classification accuracy and possession accuracy were
not measured because the example video has no corresponding ground truth. The
implemented tracking evaluator is tested on known-answer synthetic sequences.
