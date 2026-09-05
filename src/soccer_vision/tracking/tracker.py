"""Class-separated ByteTrack adapter for the pinned Ultralytics API."""

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from soccer_vision.detection import Detection
from soccer_vision.tracking.ball import BallTracker


@dataclass
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    team: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])


class SoccerTracker:
    """Keep classes separate so a ball cannot inherit a player's identity.

    Only observed tracks are emitted; missing detections never create fake ball
    positions. Public IDs stay unique across resets and scene cuts.
    """

    def __init__(self, fps: float, high: float = 0.25, low: float = 0.1,
                 buffer_seconds: float = 1, match_threshold: float = 0.8,
                 ball_max_distance: float = 100, ball_max_missing: int = 8):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.labels = ("person", "player", "goalkeeper", "referee")
        args = SimpleNamespace(track_high_thresh=high, track_low_thresh=low,
                               new_track_thresh=high, track_buffer=max(1, round(fps * buffer_seconds)),
                               match_thresh=match_threshold, fuse_score=True)
        self.trackers = {label: BYTETracker(args) for label in self.labels}
        self.ids: dict[tuple[str, int], int] = {}
        self.next_id = 1
        self.ball = BallTracker(ball_max_distance, ball_max_missing)

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()
        self.ids.clear()
        self.ball = BallTracker(self.ball.max_distance, self.ball.max_missing)

    def update(self, detections: list[Detection], shape: tuple[int, int]) -> list[Track]:
        from ultralytics.engine.results import Boxes

        output = []
        for label, tracker in self.trackers.items():
            selected = [d for d in detections if d.class_name == label]
            # ByteTrack needs low-confidence detections for its second association
            # pass. Each row is [x1,y1,x2,y2,confidence,class], in image pixels.
            rows = np.array([[d.x1, d.y1, d.x2, d.y2, d.confidence, 0] for d in selected],
                            dtype=np.float32).reshape(-1, 6)
            tracked = tracker.update(Boxes(rows, shape))
            for row in tracked:
                # Returned rows: x1,y1,x2,y2,internal_id,score,class,detection_index.
                key = (label, int(row[4]))
                if key not in self.ids:
                    self.ids[key] = self.next_id
                    self.next_id += 1
                output.append(Track(self.ids[key], label, float(row[5]), tuple(map(float, row[:4]))))
        ball = self.ball.update([d for d in detections if d.class_name == "ball"])
        if ball:
            identity, detection = ball
            key = ("ball", identity)
            if key not in self.ids:
                self.ids[key] = self.next_id
                self.next_id += 1
            output.append(Track(self.ids[key], "ball", detection.confidence,
                                (detection.x1, detection.y1, detection.x2, detection.y2)))
        return output
