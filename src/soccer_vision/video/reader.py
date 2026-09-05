"""OpenCV video reader with explicit resource ownership."""

import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Read BGR frames sequentially without loading the whole video into memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self.close()
            raise ValueError(f"Cannot open input video: {self.path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(self.fps) or self.fps <= 0 or min(self.width, self.height) <= 0:
            self.close()
            raise ValueError(f"Invalid video metadata: {self.path}")

    def frames(self) -> Iterator[np.ndarray]:
        """Yield uint8 BGR frames until EOF; OpenCV may also stop on decode errors."""
        if not self._capture.isOpened():
            raise RuntimeError("Video reader is closed")
        while True:
            success, frame = self._capture.read()
            if not success:
                break
            yield frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
