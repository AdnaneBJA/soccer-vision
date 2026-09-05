"""Reproducible confidence sweep: cache predictions once, compare precision/recall at IoU .5."""

import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import yaml
from soccer_vision.detection import SoccerDetector
from soccer_vision.evaluation.dataset import inspect_dataset
from soccer_vision.evaluation.matching import match_boxes
from soccer_vision.pipeline import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/soccer/data.yaml"))
    parser.add_argument("--model", default="models/checkpoints/best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[.2, .3, .4, .5, .6])
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/confidence.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0.001 <= value <= 1 for value in args.thresholds):
        raise ValueError("Thresholds must be between .001 and 1")
    inspect_dataset(args.data)
    data = yaml.safe_load(args.data.read_text())
    root = Path(data["path"])
    if not root.is_absolute():
        root = args.data.parent / root
    names = data["names"]
    detector = SoccerDetector(args.model, args.device, confidence=.001)
    cache = []
    for path in sorted((root / data[args.split]).glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        height, width = image.shape[:2]
        labels = path.parent.parent / "labels" / (path.stem + ".txt")
        truth = []
        for line in labels.read_text().splitlines():
            class_id, x, y, w, h = map(float, line.split())
            truth.append({"class": names[int(class_id)], "bbox": [(x-w/2)*width, (y-h/2)*height,
                                                                     (x+w/2)*width, (y+h/2)*height]})
        predictions = [{"class": d.class_name, "score": d.confidence, "bbox": [d.x1, d.y1, d.x2, d.y2]}
                       for d in detector.predict(image)]
        cache.append({"image": path.name, "truth": truth, "predictions": predictions})
    results = []
    for threshold in args.thresholds:
        counts = {name: Counter() for name in names}
        for item in cache:
            for name in names:
                truth = [row["bbox"] for row in item["truth"] if row["class"] == name]
                predicted = [row["bbox"] for row in item["predictions"] if row["class"] == name and row["score"] >= threshold]
                matched = len(match_boxes(np.array(truth), np.array(predicted)))
                counts[name].update(tp=matched, fp=len(predicted)-matched, fn=len(truth)-matched)
        rows = {}
        for name, count in counts.items():
            rows[name] = {**count, "precision": count["tp"] / (count["tp"] + count["fp"]) if count["tp"] + count["fp"] else None,
                          "recall": count["tp"] / (count["tp"] + count["fn"]) if count["tp"] + count["fn"] else None}
        results.append({"threshold": threshold, "classes": rows})
    write_json(args.output, {"model": args.model, "split": args.split, "images": len(cache),
                            "iou": .5, "results": results, "method": "Class-specific Hungarian matching of cached predictions; not mAP"})
    write_json(args.output.with_name(args.output.stem + "_predictions.json"), cache)


if __name__ == "__main__":
    main()
