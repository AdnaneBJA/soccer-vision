"""Command line entry point for the first video pipeline milestone."""

import argparse
import logging
from pathlib import Path

from soccer_vision.video import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


def copy_video(input_path: Path, output_path: Path) -> int:
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
            writer.write(first)
            count = 1
            for frame in frames:
                writer.write(frame)
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
    parser = argparse.ArgumentParser(description="Copy a video through the SoccerVision pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    try:
        copy_video(args.input, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
