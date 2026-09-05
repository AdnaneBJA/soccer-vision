"""Exercise real CPU inference on Ultralytics' bundled bus image, not soccer evaluation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import torch
import ultralytics
from ultralytics.utils import ASSETS

from soccer_vision.cli import copy_video
from soccer_vision.detection import SoccerDetector
from soccer_vision.video import VideoReader, VideoWriter
from soccer_vision.visualization.annotator import annotate


def main() -> None:
    folder = Path("outputs/smoke")
    folder.mkdir(parents=True, exist_ok=True)
    # Use a unique run directory so repeated checks never overwrite previous outputs.
    import tempfile
    folder = Path(tempfile.mkdtemp(prefix="detection-", dir=folder))
    frame = cv2.imread(str(ASSETS / "bus.jpg"))
    if frame is None:
        raise RuntimeError("Ultralytics bundled test image is missing")
    detector = SoccerDetector(device="cpu")
    detections = detector.predict(frame)
    if not any(item.class_name == "person" for item in detections):
        raise RuntimeError("Smoke image produced no person detections")
    source, target = folder / "input.mp4", folder / "annotated.mp4"
    with VideoWriter(source, 25, frame.shape[1], frame.shape[0]) as writer:
        for _ in range(3):
            writer.write(frame)
    count = copy_video(source, target, lambda image: annotate(image, detector.predict(image)))
    with VideoReader(target) as reader:
        decoded = sum(1 for _ in reader.frames())
        if decoded != 3 or reader.fps != 25:
            raise RuntimeError("Annotated video failed round-trip verification")
    report = {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
              "opencv": cv2.__version__, "device": detector.device,
              "cuda_available": torch.cuda.is_available(), "frames": count,
              "image_detections": len(detections), "output": str(target),
              "scope": "Bundled bus image smoke test; not soccer accuracy evaluation"}
    (folder / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
