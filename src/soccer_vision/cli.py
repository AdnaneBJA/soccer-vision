"""Command line entry point for copying or annotating video."""

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import cv2
import yaml

from soccer_vision.video import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


def copy_video(input_path: Path, output_path: Path,
               transform: Callable[[np.ndarray], np.ndarray] | None = None) -> int:
    """Re-encode a video frame by frame and return the number of written frames."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    with VideoReader(input_path) as reader:
        logger.info("Input: %dx%d @ %.3f FPS, %d frames", reader.width,
                    reader.height, reader.fps, reader.frame_count)
        frames = reader.frames()
        first = next(frames, None)
        if first is None:
            raise ValueError("Input video contains no decodable frames")
        with VideoWriter(output_path, reader.fps, reader.width, reader.height) as writer:
            writer.write(transform(first) if transform else first)
            count = 1
            for frame in frames:
                writer.write(transform(frame) if transform else frame)
                count += 1
                if count % 300 == 0:
                    logger.info("Processed %d frames", count)
        if reader.frame_count > 0 and count != reader.frame_count:
            raise ValueError(
                f"Decoded {count} of {reader.frame_count} reported frames; output may be incomplete"
            )
    logger.info("Saved %d frames to %s", count, output_path)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy or detect objects in a video with SoccerVision")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("copy", "detect", "analyze"), default="analyze")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--artifacts", type=Path, help="New directory for CSV, JSON and plots")
    parser.add_argument("--model", help="Pretrained default in detect mode; soccer checkpoint in analyze mode")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--calibration", type=Path)
    for name in ("tracking", "team-classification", "possession", "heatmaps"):
        parser.add_argument("--enable-" + name, action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-fps", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        if args.mode == "analyze":
            from soccer_vision.config import load_config
            from soccer_vision.pipeline import run_pipeline
            config = load_config(args.config)
            for name in ("model", "device", "confidence", "image_size", "max_frames", "show_fps"):
                if getattr(args, name) is not None:
                    setattr(config, name, getattr(args, name))
            for argument, field in (("enable_tracking", "tracking"), ("enable_team_classification", "teams"),
                                    ("enable_possession", "possession"), ("enable_heatmaps", "heatmaps")):
                if getattr(args, argument) is not None:
                    setattr(config, field, getattr(args, argument))
            if args.calibration:
                config.calibration = str(args.calibration)
            run_pipeline(args.input, args.output, config, args.artifacts)
            return 0
        transform = None
        if args.mode == "detect":
            # Fail on bad paths before loading/downloading a model.
            if not args.input.is_file():
                raise FileNotFoundError(f"Input video does not exist: {args.input}")
            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            from soccer_vision.detection import SoccerDetector
            from soccer_vision.visualization.annotator import annotate

            detector = SoccerDetector(args.model or "yolo26n.pt", args.device,
                                      args.confidence if args.confidence is not None else .35,
                                      args.image_size or 640)
            transform = lambda frame: annotate(frame, detector.predict(frame))
        copy_video(args.input, args.output, transform)
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
