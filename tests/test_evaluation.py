"""Known matching/evaluation answers and numeric report serialization."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from soccer_vision.evaluation.matching import iou_matrix, match_boxes
from soccer_vision.pipeline import write_json


class EvaluationTests(unittest.TestCase):
    def test_iou_assignment(self):
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        np.testing.assert_array_equal(iou_matrix(boxes, boxes), np.eye(2))
        self.assertEqual(match_boxes(boxes, boxes[::-1]), [(0, 1), (1, 0)])
        self.assertEqual(match_boxes([], boxes), [])

    def test_numpy_metrics_are_serializable(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "metrics.json"
            write_json(target, {"images": np.int64(25), "map": np.float64(.5), "array": np.array([1, 2])})
            self.assertEqual(json.loads(target.read_text()), {"images": 25, "map": .5, "array": [1, 2]})

    @unittest.skipUnless(importlib.util.find_spec("motmetrics"), "Optional MOT evaluation dependencies are not installed")
    def test_mot_identity_switch(self):
        from scripts.evaluate_tracking import evaluate
        truth = {frame: [{"id": 1, "class": "player", "bbox": [0, 0, 10, 10]}] for frame in range(3)}
        perfect = evaluate(truth, truth, 0, 2)
        self.assertEqual(perfect["mota"], 1)
        self.assertEqual(perfect["idf1"], 1)
        predicted = {frame: [{"id": 10 if frame < 2 else 20, "class": "player", "bbox": [0, 0, 10, 10]}] for frame in range(3)}
        switched = evaluate(truth, predicted, 0, 2)
        self.assertEqual(switched["num_switches"], 1)
        self.assertAlmostEqual(switched["mota"], 2 / 3)
        self.assertAlmostEqual(switched["idf1"], 2 / 3)
