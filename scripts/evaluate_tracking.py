"""Evaluate tracks.csv against matching fully annotated CSV frames using MOT metrics."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from soccer_vision.evaluation.matching import iou_matrix
from soccer_vision.pipeline import write_json


def load_tracks(path: Path) -> dict:
    frames = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["frame"])
            identity = int(row["track_id"])
            bbox = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"Invalid box in {path}, frame {frame}")
            if identity in [item["id"] for item in frames[frame]]:
                raise ValueError(f"Duplicate identity in {path}, frame {frame}")
            frames[frame].append({"id": identity, "class": row["class"], "bbox": bbox})
    return frames


def evaluate(truth: dict, predictions: dict, start: int, end: int, threshold: float = .5) -> dict:
    import motmetrics as mm
    mm.lap.default_solver = "scipy"
    accumulator = mm.MOTAccumulator(auto_id=True)
    for frame in range(start, end + 1):
        gt, pred = truth.get(frame, []), predictions.get(frame, [])
        distances = 1 - iou_matrix([item["bbox"] for item in gt], [item["bbox"] for item in pred])
        distances[distances > 1 - threshold] = np.nan
        for i, a in enumerate(gt):
            for j, b in enumerate(pred):
                if a["class"] != b["class"]:
                    distances[i, j] = np.nan
        accumulator.update([item["id"] for item in gt], [item["id"] for item in pred], distances)
    summary = mm.metrics.create().compute(accumulator, metrics=["num_frames", "mota", "idf1", "num_switches", "precision", "recall"], name="sequence")
    return {key: float(value) if np.isfinite(value) else None for key, value in summary.loc["sequence"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/tracking_evaluation.json"))
    args = parser.parse_args()
    if args.start_frame < 0 or args.end_frame < args.start_frame or not 0 < args.iou <= 1:
        parser.error("Invalid frame interval or IoU")
    if args.output.exists():
        raise FileExistsError(args.output)
    truth, predictions = load_tracks(args.truth), load_tracks(args.predictions)
    report = evaluate(truth, predictions, args.start_frame, args.end_frame, args.iou)
    write_json(args.output, {"metrics": report, "truth": str(args.truth), "predictions": str(args.predictions),
                            "frame_interval": [args.start_frame, args.end_frame], "iou": args.iou,
                            "requirement": "Every frame in the interval must be fully annotated, including empty frames"})


if __name__ == "__main__":
    main()
