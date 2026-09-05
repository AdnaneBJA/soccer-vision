"""Observed trajectories, smoothed motion and conservative possession estimates."""

from collections import Counter, defaultdict, deque
import numpy as np

from soccer_vision.analytics.geometry import Calibration, distance
from soccer_vision.tracking.tracker import Track


def find_player_in_possession(ball_position: tuple[float, float] | None,
                              tracks: list[Track], threshold: float) -> Track | None:
    if ball_position is None:
        return None
    players = [track for track in tracks if track.class_name == "player" and track.team is not None]
    if not players:
        return None
    nearest = min(players, key=lambda track: distance(ball_position, track.foot))
    return nearest if distance(ball_position, nearest.foot) <= threshold else None


class Analytics:
    def __init__(self, fps: float, shape: tuple[int, int], threshold: float = 70,
                 smoothing: float = .35, trail_length: int = 30,
                 calibration: Calibration | None = None):
        self.fps, self.shape, self.threshold = fps, shape, threshold
        self.smoothing, self.calibration = smoothing, calibration
        self.trails = defaultdict(lambda: deque(maxlen=trail_length))
        self.motion: dict[int, dict] = {}
        self.heatmaps: dict[int, np.ndarray] = {}
        self.team_heatmaps = {team: np.zeros((54, 96), dtype=int) for team in (0, 1)}
        self.controlled = Counter()
        self.frames = 0
        self.ball_frames = 0
        self.scene = 0

    def reset_scene(self) -> None:
        self.scene += 1
        self.trails.clear()
        # No real-world speed claims after the calibrated view has changed.
        self.calibration = None

    def update(self, tracks: list[Track], frame_index: int, possession: bool = True) -> dict:
        self.frames += 1
        balls = [track for track in tracks if track.class_name == "ball"]
        ball = max(balls, key=lambda track: track.confidence) if balls else None
        self.ball_frames += ball is not None
        owner = find_player_in_possession(ball.center if ball else None, tracks, self.threshold) if possession else None
        self.controlled[owner.team if owner else "unknown"] += 1
        positions = {}
        for track in tracks:
            point = track.center if track.class_name == "ball" else track.foot
            self.trails[track.track_id].append((frame_index, point))
            if track.class_name not in ("player", "goalkeeper", "person"):
                continue
            mapped = self.calibration.map(point) if self.calibration else None
            positions[track.track_id] = mapped
            state = self.motion.setdefault(track.track_id, {
                "track_id": track.track_id, "class": track.class_name, "team": track.team,
                "distance_px": 0.0, "top_speed_px_s": 0.0, "observed_seconds": 0.0,
                "distance_m": 0.0 if mapped else None, "top_speed_kmh": 0.0 if mapped else None,
                "last": None, "last_frame": None, "smooth": None, "mapped": None,
            })
            state["team"] = track.team
            # Only consecutive observed frames contribute motion. Gaps do not
            # produce invented paths, possession or interpolated ball locations.
            if state["last_frame"] == frame_index - 1:
                smooth = tuple(self.smoothing * np.array(point) + (1 - self.smoothing) * np.array(state["smooth"]))
                delta = distance(smooth, state["smooth"])
                state["distance_px"] += delta
                state["top_speed_px_s"] = max(state["top_speed_px_s"], delta * self.fps)
                state["observed_seconds"] += 1 / self.fps
                if mapped is not None and state["mapped"] is not None:
                    mapped = tuple(self.smoothing * np.array(mapped) + (1 - self.smoothing) * np.array(state["mapped"]))
                    meters = distance(mapped, state["mapped"])
                    state["distance_m"] += meters
                    state["top_speed_kmh"] = max(state["top_speed_kmh"], meters * self.fps * 3.6)
            else:
                smooth = point
            state.update(last=point, last_frame=frame_index, smooth=smooth, mapped=mapped)
            grid = self.heatmaps.setdefault(track.track_id, np.zeros((54, 96), dtype=int))
            x = int(np.clip(point[0] / self.shape[1] * 96, 0, 95))
            y = int(np.clip(point[1] / self.shape[0] * 54, 0, 53))
            grid[y, x] += 1
            if track.team is not None:
                self.team_heatmaps[track.team][y, x] += 1
        return {"owner_id": owner.track_id if owner else None, "team": owner.team if owner else None,
                "ball_visible": ball is not None, "field_positions": positions}

    def team_stats(self) -> dict:
        known = self.controlled[0] + self.controlled[1]
        return {"controlled_frames": known, "unknown_frames": self.controlled["unknown"],
                "total_frames": self.frames, "ball_observed_frames": self.ball_frames,
                "possession_percent": {str(team): 100 * self.controlled[team] / known if known else None
                                       for team in (0, 1)},
                "method": "nearest classified player's feet within image-pixel threshold; unknown excluded"}

    def player_stats(self) -> list[dict]:
        private = {"last", "last_frame", "smooth", "mapped"}
        result = []
        for state in self.motion.values():
            row = {key: value for key, value in state.items() if key not in private}
            seconds = row["observed_seconds"]
            row["average_speed_px_s"] = row["distance_px"] / seconds if seconds else None
            row["average_speed_kmh"] = row["distance_m"] / seconds * 3.6 if seconds and row["distance_m"] is not None else None
            result.append(row)
        return result
