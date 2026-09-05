"""Draw detection labels without modifying the source frame."""

import cv2
import numpy as np

from soccer_vision.detection import Detection

COLORS = {"person": (255, 180, 0), "player": (255, 180, 0), "ball": (0, 255, 255),
          "goalkeeper": (255, 0, 255), "referee": (0, 165, 255)}


def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy with clipped boxes, class labels, and confidence scores."""
    output = frame.copy()
    height, width = frame.shape[:2]
    for detection in detections:
        x1, x2 = [int(np.clip(value, 0, width - 1)) for value in (detection.x1, detection.x2)]
        y1, y2 = [int(np.clip(value, 0, height - 1)) for value in (detection.y1, detection.y2)]
        color = COLORS.get(detection.class_name, (255, 255, 255))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"{detection.class_name} {detection.confidence:.2f}",
                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if detection.class_name == "ball":
            cv2.circle(output, ((x1 + x2) // 2, (y1 + y2) // 2),
                       max(6, (x2 - x1) // 2 + 3), color, 2)
    return output
