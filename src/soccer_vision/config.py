"""Validated pipeline settings; YAML keys correspond to dataclass fields."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
import yaml


@dataclass
class PipelineConfig:
    model: str = "models/checkpoints/best.pt"
    device: str = "auto"
    confidence: float = 0.1
    image_size: int = 640
    tracking: bool = True
    teams: bool = True
    possession: bool = True
    heatmaps: bool = True
    show_fps: bool = True
    track_high: float = 0.25
    track_low: float = 0.1
    track_buffer_seconds: float = 1.0
    match_threshold: float = 0.8
    ball_max_distance: float = 100.0
    ball_max_missing: int = 8
    possession_threshold: float = 70.0
    team_samples: int = 60
    color_space: str = "lab"
    scene_cut_threshold: float = 0.55
    trail_length: int = 30
    smoothing: float = 0.35
    calibration: str | None = None
    max_frames: int | None = None

    def validate(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty checkpoint name/path")
        if not isinstance(self.device, str):
            raise ValueError("device must be a string")
        if self.calibration is not None and not isinstance(self.calibration, str):
            raise ValueError("calibration must be a file path")
        for name in ("confidence", "track_high", "track_low", "match_threshold",
                     "scene_cut_threshold", "smoothing"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 < self.track_low <= self.track_high or (self.tracking and self.confidence > self.track_low):
            raise ValueError("Require confidence <= track_low <= track_high, with track_low > 0")
        for name in ("image_size", "team_samples", "trail_length", "ball_max_missing"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % 32 or self.team_samples < 2:
            raise ValueError("image_size must be divisible by 32 and team_samples >= 2")
        for name in ("track_buffer_seconds", "possession_threshold", "ball_max_distance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.color_space not in ("lab", "rgb", "hsv"):
            raise ValueError("color_space must be lab, rgb or hsv")
        if self.max_frames is not None and (type(self.max_frames) is not int or self.max_frames <= 0):
            raise ValueError("max_frames must be a positive integer")
        for name in ("tracking", "teams", "possession", "heatmaps", "show_fps"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        if not self.tracking and (self.teams or self.possession or self.heatmaps):
            raise ValueError("Teams, possession and heatmaps require tracking")
        if self.possession and not self.teams:
            raise ValueError("Team possession requires team classification")


def load_config(path: str | Path | None = None) -> PipelineConfig:
    values = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(values) - {field.name for field in fields(PipelineConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return PipelineConfig(**values)
