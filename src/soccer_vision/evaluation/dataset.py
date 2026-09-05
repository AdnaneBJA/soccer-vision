"""Validate local YOLO datasets before invoking training or evaluation."""

import hashlib
from pathlib import Path
import numpy as np
import yaml


def inspect_dataset(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if names != ["ball", "goalkeeper", "player", "referee"]:
        raise ValueError("Dataset class order must be ball, goalkeeper, player, referee")
    root = Path(data["path"]) if data.get("path") else path.parent.resolve()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    counts, hashes, overlaps = {}, {}, []
    for split in ("train", "val", "test"):
        folder = root / data[split]
        images = sorted(file for file in folder.glob("*") if file.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not images:
            raise ValueError(f"No images for {split}: {folder}")
        instances = [0] * 4
        for image in images:
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                overlaps.append({"image": image.name, "splits": [hashes[digest], split]})
            hashes[digest] = split
            for line in label.read_text().splitlines():
                values = np.array([float(value) for value in line.split()])
                if len(values) != 5 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid YOLO label: {label}")
                class_id = int(values[0])
                if values[0] != class_id or not 0 <= class_id < 4 or np.any(values[1:] < 0) or np.any(values[1:] > 1) or np.any(values[3:] <= 0):
                    raise ValueError(f"Invalid YOLO box: {label}")
                instances[class_id] += 1
        counts[split] = {"images": len(images), "instances": dict(zip(names, instances))}
    return {"splits": counts, "exact_cross_split_duplicates": overlaps,
            "limitation": "Exact hashes do not detect near-duplicate frames or overlap with third-party model training"}
