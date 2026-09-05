"""IoU matching for confidence sweeps and tracking evaluation."""

import numpy as np
import lap


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(0, right - left), axis=2)
    areas_a = np.prod(np.maximum(0, a[:, 2:] - a[:, :2]), axis=1)
    areas_b = np.prod(np.maximum(0, b[:, 2:] - b[:, :2]), axis=1)
    union = areas_a[:, None] + areas_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_boxes(a: np.ndarray, b: np.ndarray, threshold: float = .5) -> list[tuple[int, int]]:
    overlap = iou_matrix(a, b)
    if not overlap.size:
        return []
    cost = 1 - overlap
    cost[overlap < threshold] = 1e6
    _, assignment, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1 - threshold + 1e-9)
    return [(index, int(target)) for index, target in enumerate(assignment) if target >= 0]
