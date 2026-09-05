"""Translate Ultralytics predictions into small, model-independent objects."""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "yolo26n.pt"
CLASS_NAMES = {"person": "person", "sports ball": "ball", "ball": "ball",
               "player": "player", "goalkeeper": "goalkeeper", "referee": "referee"}


@dataclass(frozen=True)
class Detection:
    """Bounding box in original-image pixel coordinates (left, top, right, bottom)."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def get_device(requested: str | None = None) -> str:
    """Select CUDA when available, or validate an explicit CPU/CUDA request."""
    import torch

    if requested in (None, "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda:") and requested[5:].isdigit():
        if torch.cuda.is_available() and int(requested[5:]) < torch.cuda.device_count():
            return requested
        raise ValueError(f"Requested CUDA device is unavailable: {requested}")
    raise ValueError("Device must be auto, cpu, cuda, or cuda:N")


class SoccerDetector:
    """Detect people/balls, or soccer classes from a local fine-tuned checkpoint."""

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str | None = None,
                 confidence: float = 0.35, image_size: int = 640):
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if image_size <= 0 or image_size % 32:
            raise ValueError("Image size must be a positive multiple of 32")
        if model_path != DEFAULT_MODEL and not Path(model_path).is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
        try:
            from ultralytics import YOLO
            self.device = get_device(device)
        except ImportError as exc:
            raise RuntimeError('Detection requires: python -m pip install -e ".[detection]"') from exc
        self.confidence = confidence
        self.image_size = image_size
        logger.info("Loading model %s on %s", model_path, self.device)
        # Only the documented default may be downloaded automatically.
        self.model = YOLO(model_path)
        if self.model.task != "detect":
            raise ValueError("Checkpoint must be an object detection model")
        self.class_ids = [key for key, name in self.model.names.items() if name in CLASS_NAMES]
        if not self.class_ids:
            raise ValueError("Checkpoint has no supported person/ball/soccer classes")
        if "person" in self.model.names.values():
            logger.warning("Generic person labels cannot distinguish player, referee or goalkeeper")

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Detector input must be a nonempty uint8 BGR image")
        # Ultralytics accepts H×W×3 BGR arrays, resizes/letterboxes and converts them
        # to batched RGB tensors on the chosen device. Returned boxes use original pixels.
        result = self.model.predict(source=frame, device=self.device, conf=self.confidence,
                                    imgsz=self.image_size, classes=self.class_ids, verbose=False)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        # Bring GPU tensors to CPU before converting to NumPy for the OpenCV pipeline.
        coordinates = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        detections = []
        for bbox, score, class_id in zip(coordinates, scores, classes):
            label = CLASS_NAMES.get(result.names[int(class_id)])
            if label is not None and float(score) >= self.confidence:
                detections.append(Detection(label, float(score), *map(float, bbox)))
        return detections
