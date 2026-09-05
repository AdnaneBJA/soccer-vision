"""Team colors, observed track trails, possession and performance overlays."""

import cv2
import numpy as np
from soccer_vision.tracking.tracker import Track
from soccer_vision.visualization.annotator import COLORS

TEAM_COLORS = {0: (255, 150, 20), 1: (80, 80, 255)}


def draw_tracks(frame: np.ndarray, tracks: list[Track], trails: dict, frame_index: int,
                stats: dict, fps: float | None = None) -> np.ndarray:
    result = frame.copy()
    for track in tracks:
        color = TEAM_COLORS.get(track.team, COLORS.get(track.class_name, (255, 255, 255)))
        x1, y1, x2, y2 = map(int, track.bbox)
        x1, x2 = np.clip([x1, x2], 0, frame.shape[1] - 1).tolist()
        y1, y2 = np.clip([y1, y2], 0, frame.shape[0] - 1).tolist()
        label = f"{track.class_name} #{track.track_id}" + (f" T{track.team + 1}" if track.team is not None else "")
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        # One label per track keeps crowded midfield scenes readable.
        cv2.putText(result, label, (x1, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, (x1, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        if track.class_name == "ball":
            cv2.circle(result, tuple(map(int, track.center)), max(8, (x2 - x1) // 2 + 3), color, 2)
        points = list(trails.get(track.track_id, []))
        for (previous_frame, previous), (current_frame, current) in zip(points, points[1:]):
            if current_frame - previous_frame == 1:
                cv2.line(result, tuple(map(int, previous)), tuple(map(int, current)), color, 2)
    percentages = stats["possession_percent"]
    values = [f"T{team + 1}: {percentages[str(team)]:.1f}%" if percentages[str(team)] is not None
              else f"T{team + 1}: unknown" for team in (0, 1)]
    text = "Possession " + "  ".join(values) + f" | Unknown: {stats['unknown_frames']} frames"
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 850), 55), (25, 25, 25), -1)
    cv2.putText(result, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    detail = f"Frame {frame_index} | image-space analytics"
    if fps is not None:
        detail += f" | processing {fps:.1f} FPS"
    cv2.putText(result, detail, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return result
