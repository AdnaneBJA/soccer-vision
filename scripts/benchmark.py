"""Benchmark steady-state inference on the same decoded frames across models/devices."""

import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from soccer_vision.detection import SoccerDetector
from soccer_vision.video import VideoReader
from soccer_vision.pipeline import latency_summary, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["models/checkpoints/best.pt"])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/benchmark.json"))
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup <= 0:
        parser.error("frames and warmup must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = []
    for model in args.models:
        for device in args.devices:
            detector = SoccerDetector(model, device, confidence=.1, image_size=args.image_size)
            with VideoReader(args.input) as reader:
                iterator = reader.frames()
                first = next(iterator, None)
                if first is None:
                    raise ValueError("No decodable frames")
                for _ in range(args.warmup):
                    detector.predict(first)
                if detector.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(detector.device)
                latencies = []
                for index in range(args.frames):
                    frame = first if index == 0 else next(iterator, None)
                    if frame is None:
                        break
                    start = time.perf_counter()
                    detector.predict(frame)
                    latencies.append(time.perf_counter() - start)
            reports.append({**latency_summary(latencies), "model": model, "device": detector.device,
                            "image_size": args.image_size, "warmup_frames": args.warmup,
                            "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(detector.device) / 1024**2
                            if detector.device.startswith("cuda") else None})
    write_json(args.output, {"input": str(args.input), "torch": torch.__version__, "runs": reports,
                            "method": "Same initial frames; decode excluded; predict includes preprocess, inference, postprocess and CPU transfer"})


if __name__ == "__main__":
    main()
