"""Fine-tune pretrained YOLO; persist actual training metrics and checkpoints."""

import argparse
import csv
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml
from ultralytics import YOLO
from soccer_vision.detection.detector import get_device
from soccer_vision.evaluation.dataset import inspect_dataset
from soccer_vision.pipeline import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device:
        config["device"] = args.device
    if args.name:
        config["name"] = args.name
    report = inspect_dataset(Path(config["data"]))
    if report["exact_cross_split_duplicates"]:
        raise ValueError("Dataset has exact cross-split duplicates; remove leakage before training")
    model_path = config.pop("model")
    config["device"] = get_device(config.get("device"))
    target = Path(config["project"]) / config["name"]
    if target.exists():
        raise FileExistsError(f"Use a new training run name: {target}")
    config["project"] = str(Path(config["project"]).resolve())
    model = YOLO(model_path)
    # Ultralytics builds [N,3,H,W] batches, moves weights/batches to device,
    # computes detection losses, backpropagates, and updates pretrained weights.
    model.train(**config)
    run = model.trainer.save_dir
    with model.trainer.csv.open(newline="", encoding="utf-8") as stream:
        epochs = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    write_json(run / "training_metrics.json", {"dataset": report, "epochs": epochs,
                                               "base_model": model_path, "config": config})
    destination = Path("models/checkpoints") / (config["name"] + ".pt")
    if destination.exists():
        raise FileExistsError(f"Checkpoint destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model.trainer.best, destination)
    print(f"Training complete: {destination}; metrics/curves: {run}")


if __name__ == "__main__":
    main()
