"""Exercise all artifact writers with deterministic injected soccer detections."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from soccer_vision.config import PipelineConfig
from soccer_vision.detection import Detection
from soccer_vision.pipeline import run_pipeline
from soccer_vision.video import VideoReader, VideoWriter


class FakeDetector:
    device = "cpu"

    def predict(self, frame):
        return [Detection("player", .9, 10, 20, 30, 80), Detection("player", .9, 60, 20, 80, 80),
                Detection("ball", .9, 17, 75, 23, 81), Detection("referee", .9, 110, 20, 130, 80)]


class PipelineTests(unittest.TestCase):
    def test_complete_artifact_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target, artifacts = root / "in.mp4", root / "out.mp4", root / "artifacts"
            frame = np.full((100, 160, 3), (0, 130, 0), dtype=np.uint8)
            frame[20:80, 10:30] = (230, 30, 20)
            frame[20:80, 60:80] = (20, 30, 230)
            with VideoWriter(source, 25, 160, 100) as writer:
                for _ in range(8):
                    writer.write(frame)
            report = run_pipeline(source, target, PipelineConfig(team_samples=4), artifacts, FakeDetector())
            self.assertEqual(report["processed_frames"], 8)
            self.assertTrue(report["team_classifier_fitted"])
            with (artifacts / "tracks.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 32)
            self.assertEqual(len({row["track_id"] for row in rows}), 4)
            stats = json.loads((artifacts / "team_stats.json").read_text())
            self.assertGreater(stats["controlled_frames"], 0)
            self.assertTrue((artifacts / "plots/team_1_heatmap.png").is_file())
            self.assertTrue((artifacts / "report.html").is_file())
            self.assertEqual(json.loads((artifacts / "status.json").read_text())["status"], "complete")
            with VideoReader(target) as reader:
                self.assertEqual(sum(1 for _ in reader.frames()), 8)
            with self.assertRaises(FileExistsError):
                run_pipeline(source, target, PipelineConfig(), artifacts, FakeDetector())
