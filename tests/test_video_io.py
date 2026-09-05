"""Exercise real MP4 encoding/decoding using a small synthetic clip."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_vision.cli import copy_video, main
from soccer_vision.video import VideoReader, VideoWriter


class VideoIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_clip(self):
        path = self.root / "input.mp4"
        with VideoWriter(path, 25.0, 96, 64) as writer:
            for index in range(20):
                writer.write(np.full((64, 96, 3), index * 10, dtype=np.uint8))
        return path

    def test_round_trip_preserves_timing_dimensions_and_content(self):
        source = self.make_clip()
        target = self.root / "nested" / "copy.mp4"
        self.assertEqual(copy_video(source, target), 20)
        with VideoReader(source) as original, VideoReader(target) as result:
            self.assertEqual((result.width, result.height, result.frame_count), (96, 64, 20))
            self.assertAlmostEqual(result.fps, original.fps, places=3)
            self.assertAlmostEqual(result.frame_count / result.fps, 0.8, places=3)
            before, after = list(original.frames()), list(result.frames())
            self.assertEqual(len(after), 20)
            # MP4 is lossy: allow small codec changes, while checking frame ordering.
            for expected, actual in zip(before, after):
                self.assertLess(np.abs(expected.astype(float) - actual).mean(), 10)

    def test_missing_and_corrupt_input(self):
        with self.assertRaises(FileNotFoundError):
            VideoReader(self.root / "missing.mp4")
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not a video")
        with self.assertRaises(ValueError):
            VideoReader(corrupt)

    def test_output_protection(self):
        source = self.make_clip()
        saved = source.read_bytes()
        with self.assertRaises(ValueError):
            copy_video(source, source)
        target = self.root / "existing.mp4"
        target.write_bytes(b"existing output")
        with self.assertRaises(FileExistsError):
            copy_video(source, target)
        self.assertEqual(target.read_bytes(), b"existing output")
        self.assertEqual(source.read_bytes(), saved)

    def test_writer_validation_and_closed_resources(self):
        for fps, width in [(0, 96), (float("nan"), 96), (25, 95)]:
            with self.assertRaises(ValueError):
                VideoWriter(self.root / "bad.mp4", fps, width, 64)
        with VideoWriter(self.root / "valid.mp4", 25, 96, 64) as writer:
            with self.assertRaises(ValueError):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                writer.write(np.zeros((64, 96, 3), dtype=float))
        with self.assertRaises(RuntimeError):
            writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
        with VideoReader(self.make_clip()) as reader:
            pass
        with self.assertRaises(RuntimeError):
            next(reader.frames())

    def test_cli_success_and_failure(self):
        self.assertEqual(main(["--mode", "copy", "--input", str(self.make_clip()), "--output",
                               str(self.root / "cli.mp4")]), 0)
        self.assertEqual(main(["--mode", "copy", "--input", str(self.root / "absent.mp4"), "--output",
                               str(self.root / "unused.mp4")]), 1)


if __name__ == "__main__":
    unittest.main()
