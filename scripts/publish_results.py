"""Copy measured summaries into Git-friendly reports, removing workspace-specific paths."""

import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soccer_vision.pipeline import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=Path("outputs/complete_demo"))
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/latest"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = str(Path.cwd())
    def clean(value):
        if isinstance(value, str):
            return value.replace(root, ".").replace("\\", "/")
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value
    inputs = {"demo_benchmark.json": args.demo / "benchmark.json", "team_stats.json": args.demo / "team_stats.json",
              "reference_evaluation.json": Path("outputs/evaluation_reference/evaluation.json"),
              "local_model_evaluation.json": Path("outputs/evaluation_cpu3/evaluation.json"),
              "confidence_experiment.json": Path("outputs/experiments/confidence.json"),
              "inference_benchmark.json": Path("outputs/metrics/benchmark.json"),
              "training_metrics.json": args.training / "training_metrics.json",
              "reference_model_source.json": Path("models/checkpoints/best.pt.source.json"),
              "demo_source.json": Path("samples/input/match.mp4.source.json")}
    documents = {name: clean(json.loads(path.read_text(encoding="utf-8"))) for name, path in inputs.items()}
    manifest = json.loads(Path("data/soccer/manifest.json").read_text(encoding="utf-8"))
    documents["dataset_source.json"] = {key: value for key, value in manifest.items() if key != "files"}
    documents["dataset_source.json"]["downloaded_files"] = len(manifest["files"])
    for name, document in documents.items():
        write_json(args.output / name, document)
    for source, destination in ((args.training / "results.png", "training_curves.png"),
                                (Path("outputs/evaluation_reference/confusion_matrix_normalized.png"), "confusion_matrix.png")):
        if source.exists():
            shutil.copy2(source, args.output / destination)
    print(f"Published measured summaries to {args.output}")


if __name__ == "__main__":
    main()
