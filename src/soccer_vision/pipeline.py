"""End-to-end streaming soccer tracking, analytics and artifact export."""

from dataclasses import asdict
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from soccer_vision.config import PipelineConfig
from soccer_vision.detection import SoccerDetector
from soccer_vision.video import VideoReader, VideoWriter
from soccer_vision.tracking.tracker import SoccerTracker
from soccer_vision.teams.classifier import TeamClassifier
from soccer_vision.analytics.engine import Analytics
from soccer_vision.analytics.geometry import Calibration
from soccer_vision.analytics.heatmaps import save_heatmap
from soccer_vision.visualization.annotator import annotate
from soccer_vision.visualization.overlays import draw_tracks
from soccer_vision.visualization.report import write_report

logger = logging.getLogger(__name__)


def write_json(path: Path, value: object) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert), encoding="utf-8")


def latency_summary(latencies: list[float]) -> dict:
    if not latencies:
        return {"frames": 0, "avg_latency_ms": None, "p50_latency_ms": None,
                "p95_latency_ms": None, "inference_fps": None}
    values = np.array(latencies) * 1000
    return {"frames": len(values), "avg_latency_ms": float(values.mean()),
            "p50_latency_ms": float(np.percentile(values, 50)),
            "p95_latency_ms": float(np.percentile(values, 95)),
            "inference_fps": 1000 / float(values.mean())}


class SceneDetector:
    """Histogram-distance cut heuristic; misses similar-looking camera cuts."""

    def __init__(self, threshold: float):
        self.threshold, self.previous = threshold, None

    def update(self, frame: np.ndarray) -> bool:
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)
        cut = self.previous is not None and cv2.compareHist(self.previous, histogram, cv2.HISTCMP_BHATTACHARYYA) > self.threshold
        self.previous = histogram
        return cut


def run_pipeline(input_path: Path, output_path: Path, config: PipelineConfig,
                 artifacts: Path | None = None, detector=None) -> dict:
    """Process each frame once; export observed tracks and reproducible run metadata.

    Detector injection supports integration tests using deterministic detections.
    Production uses SoccerDetector and real PyTorch inference.
    """
    config.validate()
    input_path, output_path = Path(input_path), Path(output_path)
    artifacts = Path(artifacts) if artifacts else output_path.parent / (output_path.stem + "_artifacts")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if output_path.exists() or artifacts.exists():
        raise FileExistsError("Use a new output video and artifact directory for each run")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    start = time.perf_counter()
    calibration = Calibration.load(config.calibration) if config.calibration else None
    with VideoReader(input_path) as reader:
        detector = detector or SoccerDetector(config.model, config.device, config.confidence, config.image_size)
        tracker = SoccerTracker(reader.fps, config.track_high, config.track_low,
                                config.track_buffer_seconds, config.match_threshold,
                                config.ball_max_distance, config.ball_max_missing) if config.tracking else None
        teams = TeamClassifier(config.color_space, config.team_samples)
        analytics = Analytics(reader.fps, (reader.height, reader.width), config.possession_threshold,
                              config.smoothing, config.trail_length, calibration)
        scenes = SceneDetector(config.scene_cut_threshold)
        latencies, warmup_latency = [], None
        class_counts = Counter()
        cuts, frames = [], 0
        artifacts.mkdir(parents=True)
        write_json(artifacts / "config.json", asdict(config))
        write_json(artifacts / "status.json", {"status": "running"})
        metadata = {"fps": reader.fps, "width": reader.width, "height": reader.height,
                    "reported_frames": reader.frame_count}
        try:
            with VideoWriter(output_path, reader.fps, reader.width, reader.height) as writer, \
                    (artifacts / "tracks.csv").open("w", newline="", encoding="utf-8") as tracks_file, \
                    (artifacts / "possession.csv").open("w", newline="", encoding="utf-8") as possession_file:
                track_csv = csv.writer(tracks_file)
                track_csv.writerow(["frame", "time_s", "scene", "track_id", "class", "team", "x1", "y1", "x2", "y2",
                                    "confidence", "foot_x", "foot_y", "field_x_m", "field_y_m"])
                possession_csv = csv.writer(possession_file)
                possession_csv.writerow(["frame", "owner_id", "team", "ball_visible"])
                for frame_index, frame in enumerate(reader.frames()):
                    if config.max_frames is not None and frame_index >= config.max_frames:
                        break
                    frame_start = time.perf_counter()
                    if scenes.update(frame):
                        cuts.append(frame_index)
                        if tracker:
                            tracker.reset()
                        analytics.reset_scene()
                        teams.votes.clear()
                        logger.info("Scene cut at frame %d; reset tracks", frame_index)
                    # predict() transfers output tensors to CPU, synchronizing CUDA
                    # before timing ends. First-frame setup is reported separately.
                    inference_start = time.perf_counter()
                    detections = detector.predict(frame)
                    elapsed = time.perf_counter() - inference_start
                    if frame_index == 0:
                        warmup_latency = elapsed
                    else:
                        latencies.append(elapsed)
                    class_counts.update(d.class_name for d in detections)
                    if tracker:
                        tracks = tracker.update(detections, frame.shape[:2])
                        if config.teams:
                            for track in tracks:
                                if track.class_name == "player":
                                    x1, y1, x2, y2 = track.bbox
                                    crop = frame[max(0, int(y1)):min(reader.height, int(y2)),
                                                 max(0, int(x1)):min(reader.width, int(x2))]
                                    track.team = teams.assign(track.track_id, crop)
                        state = analytics.update(tracks, frame_index, config.possession)
                        for track in tracks:
                            mapped = state["field_positions"].get(track.track_id) or (None, None)
                            track_csv.writerow([frame_index, frame_index / reader.fps, analytics.scene,
                                                track.track_id, track.class_name, track.team, *track.bbox,
                                                track.confidence, *track.foot, *mapped])
                        possession_csv.writerow([frame_index, state["owner_id"], state["team"], state["ball_visible"]])
                        annotated = draw_tracks(frame, tracks, analytics.trails, frame_index, analytics.team_stats(),
                                                1 / (time.perf_counter() - frame_start) if config.show_fps else None)
                    else:
                        annotated = annotate(frame, detections)
                    writer.write(annotated)
                    frames += 1
                    if frames % 100 == 0:
                        logger.info("Processed %d/%d frames", frames, reader.frame_count)
                if not frames:
                    raise ValueError("No decodable input frames")
                expected = min(reader.frame_count, config.max_frames) if config.max_frames else reader.frame_count
                if expected > 0 and frames != expected:
                    raise ValueError(f"Decoded {frames}/{expected} frames; output is incomplete")
            video_seconds = time.perf_counter() - start
            write_json(artifacts / "player_stats.json", analytics.player_stats())
            write_json(artifacts / "team_stats.json", analytics.team_stats())
            if config.heatmaps:
                for team, grid in analytics.team_heatmaps.items():
                    save_heatmap(grid, artifacts / "plots" / f"team_{team + 1}_heatmap.png", f"Team {team + 1}", (reader.height, reader.width))
                for track_id, grid in analytics.heatmaps.items():
                    save_heatmap(grid, artifacts / "plots" / f"player_{track_id}_heatmap.png", f"Track {track_id}", (reader.height, reader.width))
            import torch
            import ultralytics
            report = {**latency_summary(latencies), "processed_frames": frames, "device": detector.device,
                      "warmup_latency_ms": warmup_latency * 1000, "video_processing_seconds": video_seconds,
                      "pipeline_fps": frames / video_seconds, "total_seconds": time.perf_counter() - start,
                      "input": str(input_path), "output": str(output_path), "video": metadata,
                      "model": config.model, "detections_by_class": dict(class_counts), "scene_cuts": cuts,
                      "unique_tracks": tracker.next_id - 1 if tracker else 0,
                      "team_classifier_fitted": teams.centers is not None,
                      "versions": {"python": platform.python_version(), "torch": torch.__version__,
                                   "ultralytics": ultralytics.__version__, "opencv": cv2.__version__},
                      "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(detector.device) / 1024**2
                      if detector.device.startswith("cuda") else None,
                      "timing": "Inference excludes first-frame warmup; pipeline includes initialization, decoding, annotations and video writing"}
            write_json(artifacts / "benchmark.json", report)
            write_report(artifacts, output_path, analytics.player_stats(), analytics.team_stats(), report, config.heatmaps)
            write_json(artifacts / "status.json", {"status": "complete", "frames": frames})
            logger.info("Saved %s and artifacts in %s", output_path, artifacts)
            return report
        except Exception as exc:
            write_json(artifacts / "status.json", {"status": "failed", "frames": frames, "error": str(exc)})
            raise
