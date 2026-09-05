"""Coordinate helpers and explicit fixed-camera homography."""

import json
from pathlib import Path
import cv2
import numpy as np


def bbox_bottom_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.asarray(a) - b))


class Calibration:
    """Map foot positions to meters for a fixed camera only.

    A static homography cannot compensate broadcast pan/tilt/zoom. Calibration
    is invalidated on detected cuts; moving-camera clips need per-frame mapping.
    """

    def __init__(self, image_points: list, field_points: list):
        source, target = np.asarray(image_points, dtype=float), np.asarray(field_points, dtype=float)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 4:
            raise ValueError("Calibration needs at least four matching image/field points")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("Calibration coordinates must be finite")
        self.matrix, _ = cv2.findHomography(source, target, method=0)
        if self.matrix is None or np.linalg.matrix_rank(self.matrix) < 3:
            raise ValueError("Degenerate calibration points")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if values.get("camera") != "fixed":
            raise ValueError("Static calibration requires camera: fixed")
        return cls(values["image_points"], values["field_points_m"])

    def map(self, point: tuple[float, float]) -> tuple[float, float]:
        vector = self.matrix @ np.array([*point, 1.0])
        if abs(vector[2]) < 1e-8:
            raise ValueError("Point maps to the homography horizon")
        return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))
