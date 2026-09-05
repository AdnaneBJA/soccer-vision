"""Deterministic behavior checks for tracking, teams, geometry and analytics."""

import unittest
import numpy as np

from soccer_vision.analytics.engine import Analytics, find_player_in_possession
from soccer_vision.analytics.geometry import Calibration, bbox_bottom_center, distance
from soccer_vision.config import PipelineConfig
from soccer_vision.detection import Detection
from soccer_vision.teams.classifier import TeamClassifier, color_feature
from soccer_vision.tracking.tracker import SoccerTracker, Track
from soccer_vision.tracking.ball import BallTracker
from soccer_vision.pipeline import SceneDetector, latency_summary


class GeometryTests(unittest.TestCase):
    def test_coordinates_and_homography(self):
        self.assertEqual(bbox_bottom_center((10, 20, 30, 60)), (20, 60))
        self.assertEqual(distance((0, 0), (3, 4)), 5)
        calibration = Calibration([[0, 0], [100, 0], [100, 100], [0, 100]],
                                  [[0, 0], [10, 0], [10, 10], [0, 10]])
        np.testing.assert_allclose(calibration.map((50, 50)), (5, 5))
        with self.assertRaises(ValueError):
            Calibration([[0, 0]] * 4, [[0, 0]] * 4)

    def test_config_validation(self):
        PipelineConfig().validate()
        for kwargs in ({"confidence": .4}, {"smoothing": float("nan")}, {"image_size": 10},
                       {"max_frames": 0}, {"tracking": False}, {"teams": "false"}):
            with self.assertRaises(ValueError):
                PipelineConfig(**kwargs).validate()


class TrackingTests(unittest.TestCase):
    def test_fast_ball_and_missing_observations(self):
        tracker = BallTracker(max_distance=100, max_missing=2)
        first = tracker.update([Detection("ball", .9, 0, 0, 4, 4)])
        second = tracker.update([Detection("ball", .8, 25, 0, 29, 4)])
        self.assertEqual(first[0], second[0])
        self.assertIsNone(tracker.update([]))
        recovered = tracker.update([Detection("ball", .8, 70, 0, 74, 4)])
        self.assertEqual(recovered[0], first[0])
        for _ in range(3):
            self.assertIsNone(tracker.update([]))
        new = tracker.update([Detection("ball", .8, 800, 0, 804, 4)])
        self.assertNotEqual(new[0], first[0])

    def test_identity_low_confidence_recovery_and_class_separation(self):
        tracker = SoccerTracker(25)
        first = tracker.update([Detection("player", .9, 10, 10, 30, 60),
                                Detection("ball", .9, 10, 10, 15, 15)], (100, 100))
        second = tracker.update([Detection("player", .15, 11, 10, 31, 60),
                                 Detection("ball", .9, 11, 10, 16, 15)], (100, 100))
        self.assertEqual([track.track_id for track in first], [track.track_id for track in second])
        self.assertNotEqual(first[0].track_id, first[1].track_id)
        self.assertEqual(tracker.update([], (100, 100)), [])
        recovered = tracker.update([Detection("player", .9, 12, 10, 32, 60)], (100, 100))
        self.assertEqual(recovered[0].track_id, first[0].track_id)
        tracker.reset()
        new = tracker.update([Detection("player", .9, 12, 10, 32, 60)], (100, 100))
        self.assertGreater(new[0].track_id, max(track.track_id for track in first))

    def test_scene_change(self):
        scene = SceneDetector(.55)
        green = np.full((64, 96, 3), (0, 140, 0), dtype=np.uint8)
        self.assertFalse(scene.update(green))
        self.assertFalse(scene.update(green))
        self.assertTrue(scene.update(np.full_like(green, (0, 0, 255))))


class TeamTests(unittest.TestCase):
    def test_two_jerseys_and_grass(self):
        blue = np.full((80, 40, 3), (220, 30, 20), dtype=np.uint8)
        red = np.full_like(blue, (20, 30, 220))
        green = np.full_like(blue, (0, 150, 0))
        for space in ("rgb", "hsv", "lab"):
            classifier = TeamClassifier(space, 4)
            classifier.fit([blue, blue, red, red])
            self.assertNotEqual(classifier.predict(blue), classifier.predict(red))
            self.assertIsNone(classifier.predict(green))
        self.assertIsNone(color_feature(np.empty((0, 0, 3), dtype=np.uint8)))
        with self.assertRaises(ValueError):
            TeamClassifier().fit([blue, blue, blue, blue])


class AnalyticsTests(unittest.TestCase):
    def test_possession_unknown_and_threshold(self):
        player = Track(1, "player", .9, (0, 0, 20, 40), 0)
        self.assertEqual(find_player_in_possession((10, 40), [player], 1), player)
        self.assertIsNone(find_player_in_possession((10, 42), [player], 1))
        self.assertIsNone(find_player_in_possession(None, [player], 10))
        analytics = Analytics(25, (100, 100))
        analytics.update([player, Track(2, "ball", .9, (8, 38, 12, 42))], 0)
        analytics.update([player], 1)
        stats = analytics.team_stats()
        self.assertEqual(stats["possession_percent"]["0"], 100)
        self.assertEqual(stats["unknown_frames"], 1)

    def test_motion_units_gaps_and_heatmap_counts(self):
        analytics = Analytics(10, (100, 100), smoothing=1)
        analytics.update([Track(1, "player", .9, (0, 0, 20, 40), 0)], 0)
        analytics.update([Track(1, "player", .9, (3, 0, 23, 44), 0)], 1)
        analytics.update([Track(1, "player", .9, (70, 0, 90, 80), 0)], 5)
        stats = analytics.player_stats()[0]
        self.assertEqual(stats["distance_px"], 5)
        self.assertEqual(stats["top_speed_px_s"], 50)
        self.assertIsNone(stats["distance_m"])
        self.assertEqual(analytics.team_heatmaps[0].sum(), 3)
        analytics.reset_scene()
        self.assertFalse(analytics.trails)

    def test_latency_statistics(self):
        result = latency_summary([.01, .02, .03])
        self.assertAlmostEqual(result["avg_latency_ms"], 20)
        self.assertAlmostEqual(result["inference_fps"], 50)
        self.assertIsNone(latency_summary([])["avg_latency_ms"])
