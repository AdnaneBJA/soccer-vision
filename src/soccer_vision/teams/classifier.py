"""Small deterministic two-means classifier, with grass rejection and ID voting."""

from collections import Counter
import cv2
import numpy as np


def color_feature(crop: np.ndarray, space: str = "lab") -> np.ndarray | None:
    """Extract median jersey color from the central upper torso, rejecting grass."""
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return None
    height, width = crop.shape[:2]
    jersey = crop[int(height * .2):max(int(height * .55), 2),
                  int(width * .2):max(int(width * .8), 2)]
    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    grass = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] > 60)
    pixels = jersey[~grass]
    if len(pixels) < 8:
        return None
    if space == "lab":
        converted = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return np.median(converted, axis=0) / 255
    if space == "rgb":
        return np.median(pixels[:, ::-1], axis=0) / 255
    if space != "hsv":
        raise ValueError("Unknown color space")
    values = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    angles = values[:, 0].astype(float) * (2 * np.pi / 180)
    # Circular hue avoids treating red near 0 and 179 as opposite colors.
    return np.array([np.mean(np.cos(angles)), np.mean(np.sin(angles)),
                     np.median(values[:, 1]) / 255, np.median(values[:, 2]) / 255])


class TeamClassifier:
    def __init__(self, space: str = "lab", samples: int = 60):
        self.space, self.samples = space, samples
        self.features: list[np.ndarray] = []
        self.centers: np.ndarray | None = None
        self.votes: dict[int, Counter] = {}

    def fit_features(self, features: list[np.ndarray]) -> bool:
        if len(features) < 2:
            return False
        data = np.stack(features)
        first = data[np.argmin(data[:, 0])]
        second = data[np.argmax(np.linalg.norm(data - first, axis=1))]
        centers = np.stack([first, second])
        for _ in range(30):
            labels = np.argmin(np.linalg.norm(data[:, None] - centers, axis=2), axis=1)
            if any(np.sum(labels == team) < 2 for team in range(2)):
                return False
            updated = np.stack([data[labels == team].mean(axis=0) for team in range(2)])
            if np.allclose(updated, centers):
                break
            centers = updated
        if np.linalg.norm(centers[0] - centers[1]) < .08:
            return False
        self.centers = centers
        return True

    def fit(self, player_crops: list[np.ndarray]) -> None:
        features = [color_feature(crop, self.space) for crop in player_crops]
        if not self.fit_features([feature for feature in features if feature is not None]):
            raise ValueError("Need distinguishable examples from two jersey colors")

    def predict(self, player_crop: np.ndarray) -> int | None:
        feature = color_feature(player_crop, self.space)
        if feature is None or self.centers is None:
            return None
        return int(np.argmin(np.linalg.norm(self.centers - feature, axis=1)))

    def assign(self, track_id: int, crop: np.ndarray) -> int | None:
        feature = color_feature(crop, self.space)
        if feature is None:
            return self.votes[track_id].most_common(1)[0][0] if track_id in self.votes else None
        if self.centers is None:
            self.features.append(feature)
            self.features = self.features[-self.samples:]
            if len(self.features) < self.samples or not self.fit_features(self.features):
                return None
        distances = np.linalg.norm(self.centers - feature, axis=1)
        if abs(float(distances[0] - distances[1])) < .03:
            return None
        votes = self.votes.setdefault(track_id, Counter())
        votes[int(np.argmin(distances))] += 1
        return votes.most_common(1)[0][0]
