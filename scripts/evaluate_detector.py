"""Measure class-specific mAP, precision and recall on a labeled local split."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO
from soccer_vision.detection.detector import get_device
from soccer_vision.evaluation.dataset import inspect_dataset
from soccer_vision.pipeline import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/soccer/data.yaml"))
    parser.add_argument("--model", default="models/checkpoints/best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    dataset = inspect_dataset(args.data)
    model = YOLO(args.model)
    if list(model.names.values()) != ["ball", "goalkeeper", "player", "referee"]:
        raise ValueError("Evaluation requires a four-class soccer model matching dataset label order")
    device = get_device(args.device)
    metrics = model.val(data=str(args.data.resolve()), split=args.split, device=device,
                        imgsz=args.image_size, batch=4, workers=0, plots=True, conf=.001,
                        project=str(args.output.parent.resolve()), name=args.output.name, exist_ok=False)
    write_json(args.output / "evaluation.json", {
        "model": args.model, "split": args.split, "device": device, "image_size": args.image_size,
        "dataset": dataset, "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "classes": metrics.summary(), "speed_ms": metrics.speed,
        "limitation": "Third-party checkpoint may have seen related frames; these are in-domain measurements, not independent generalization evidence"})


if __name__ == "__main__":
    main()
