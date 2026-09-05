"""Deterministic adapter tests; no weights or downloads required."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soccer_vision.detection.detector import Detection, SoccerDetector, get_device
from soccer_vision.visualization.annotator import annotate


def tensor(values):
    result = Mock()
    result.detach.return_value.cpu.return_value.numpy.return_value = np.array(values)
    return result


class DetectionTests(unittest.TestCase):
    def test_adapter_filters_and_preserves_pixel_coordinates(self):
        detector = SoccerDetector.__new__(SoccerDetector)
        detector.device, detector.confidence, detector.image_size = "cpu", 0.35, 640
        detector.class_ids = [0, 32]
        detector.model = Mock()
        detector.model.predict.return_value = [SimpleNamespace(
            names={0: "person", 32: "sports ball", 2: "car"}, boxes=SimpleNamespace(
                xyxy=tensor([[10, 20, 40, 80]] * 4), conf=tensor([0.9, 0.8, 0.1, 0.9]),
                cls=tensor([0, 32, 0, 2])))]
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        detections = detector.predict(frame)
        self.assertEqual([d.class_name for d in detections], ["person", "ball"])
        self.assertEqual(detections[0], Detection("person", 0.9, 10, 20, 40, 80))
        self.assertIs(detector.model.predict.call_args.kwargs["source"], frame)
        self.assertEqual(detector.model.predict.call_args.kwargs["device"], "cpu")
        detector.model.predict.return_value[0].boxes = None
        self.assertEqual(detector.predict(frame), [])
        with self.assertRaises(ValueError):
            detector.predict(frame.astype(float))

    def test_device_selection(self):
        torch = Mock()
        with patch.dict(sys.modules, {"torch": torch}):
            torch.cuda.is_available.return_value = False
            self.assertEqual(get_device(), "cpu")
            with self.assertRaises(ValueError):
                get_device("cuda")
            torch.cuda.is_available.return_value = True
            torch.cuda.device_count.return_value = 1
            self.assertEqual(get_device(), "cuda:0")
            self.assertEqual(get_device("cpu"), "cpu")
            for invalid in ("cuda:2", "banana", "cuda:-1"):
                with self.assertRaises(ValueError):
                    get_device(invalid)

    def test_configuration_errors_before_model_loading(self):
        with self.assertRaises(FileNotFoundError):
            SoccerDetector("nonexistent-checkpoint.pt")
        for confidence in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                SoccerDetector(confidence=confidence)
        with self.assertRaises(ValueError):
            SoccerDetector(image_size=31)

    def test_annotation_preserves_input(self):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        result = annotate(frame, [Detection("ball", 0.8, -1, 20, 30, 40)])
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(result.any())
        self.assertFalse(frame.any())
        np.testing.assert_array_equal(annotate(frame, []), frame)
