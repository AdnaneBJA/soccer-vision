"""Command line entry point for copying or annotating video."""

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np

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
    parser.add_argument("--mode", choices=("copy", "detect"), default="copy")
    parser.add_argument("--model", default="yolo26n.pt", help="Default pretrained model or local checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        transform = None
        if args.mode == "detect":
            # Fail on bad paths before loading/downloading a model.
            if not args.input.is_file():
                raise FileNotFoundError(f"Input video does not exist: {args.input}")
            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            from soccer_vision.detection import SoccerDetector
            from soccer_vision.visualization.annotator import annotate

            detector = SoccerDetector(args.model, args.device, args.confidence, args.image_size)
            transform = lambda frame: annotate(frame, detector.predict(frame))
        copy_video(args.input, args.output, transform)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
