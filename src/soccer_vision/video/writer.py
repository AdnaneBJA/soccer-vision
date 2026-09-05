"""Validated MP4 output."""

import math
from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Write BGR frames to MP4, preserving the configured frame rate and size."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("Output video must have an .mp4 extension")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("MP4 dimensions must be positive and even to avoid codec cropping")
        self.width, self.height = width, height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self._writer.isOpened():
            self.close()
            raise ValueError(f"Cannot open MP4 writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if not self._writer.isOpened():
            raise RuntimeError("Video writer is closed")
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError("Frame must be uint8 BGR with the configured height and width")
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
