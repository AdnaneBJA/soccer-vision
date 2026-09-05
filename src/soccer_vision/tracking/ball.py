"""Single-ball association for small boxes that do not overlap between frames."""

import numpy as np
from soccer_vision.detection import Detection


class BallTracker:
    """Associate observed ball centers using a constant-velocity gate.

    Missing frames update elapsed time only; predicted positions are never
    exported or counted toward possession. Long gaps start a new identity.
    """

    def __init__(self, max_distance: float = 100, max_missing: int = 8):
        self.max_distance, self.max_missing = max_distance, max_missing
        self.center: np.ndarray | None = None
        self.velocity = np.zeros(2)
        self.gap = 0
        self.identity = 0

    def update(self, detections: list[Detection]) -> tuple[int, Detection] | None:
        self.gap += 1
        if not detections:
            return None
        centers = np.array([[(d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2] for d in detections])
        if self.center is None or self.gap > self.max_missing + 1:
            index = int(np.argmax([d.confidence for d in detections]))
            self.identity += 1
            self.velocity = np.zeros(2)
        else:
            predicted = self.center + self.velocity * self.gap
            distances = np.linalg.norm(centers - predicted, axis=1)
            index = int(np.argmin(distances))
            if distances[index] > self.max_distance * min(self.gap, 3):
                return None
            velocity = (centers[index] - self.center) / self.gap
            self.velocity = .5 * self.velocity + .5 * velocity
        self.center = centers[index]
        self.gap = 0
        return self.identity, detections[index]
