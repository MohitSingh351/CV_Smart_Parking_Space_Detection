"""Full per-frame pipeline: detect vehicles, decide occupancy, draw overlay."""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from src.occupancy import Spot, compute_occupancy

OCCUPIED_COLOR = (0, 0, 255)  # red, BGR
FREE_COLOR = (0, 200, 0)  # green, BGR


@dataclass
class PipelineResult:
    annotated_frame: np.ndarray
    occupancy: dict[str, bool]

    @property
    def occupied_count(self) -> int:
        return sum(self.occupancy.values())

    @property
    def free_count(self) -> int:
        return len(self.occupancy) - self.occupied_count

    @property
    def total_count(self) -> int:
        return len(self.occupancy)


def load_spots(spots_json_path: str) -> list[Spot]:
    with open(spots_json_path) as f:
        data = json.load(f)
    return [Spot.from_points(entry["id"], entry["points"]) for entry in data["spots"]]


def _draw_overlay(frame: np.ndarray, spots: list[Spot], occupancy: dict[str, bool]) -> np.ndarray:
    annotated = frame.copy()
    overlay = frame.copy()
    for spot in spots:
        color = OCCUPIED_COLOR if occupancy[spot.id] else FREE_COLOR
        points = np.array(spot.polygon.exterior.coords, dtype=int)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)

    annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

    occupied = sum(occupancy.values())
    total = len(occupancy)
    label = f"Occupied: {occupied}/{total}  Free: {total - occupied}/{total}"
    cv2.rectangle(annotated, (0, 0), (420, 40), (0, 0, 0), -1)
    cv2.putText(annotated, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


METHODS = ("yolo", "classifier")


class ParkingPipeline:
    """Decides occupancy for one camera view and draws the overlay.

    Methods:
        yolo       - detect vehicles with YOLO, a spot is occupied if a box covers enough of it
        classifier - per-spot CNN answering "empty or occupied?" on each spot crop
    """

    def __init__(
        self,
        spots: str | list[Spot],
        method: str = "yolo",
        model_path: str | None = None,
        overlap_threshold: float = 0.3,
        confidence: float = 0.35,
        probability_threshold: float = 0.5,
    ):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")
        self.spots = load_spots(spots) if isinstance(spots, str) else spots
        self.method = method
        self.overlap_threshold = overlap_threshold
        self.probability_threshold = probability_threshold
        # Import model code lazily so each method only loads what it needs.
        if method == "yolo":
            from src.detector import VehicleDetector

            self.detector = VehicleDetector(model_path=model_path or "yolov8n.pt", confidence=confidence)
        elif method == "classifier":
            from src.spot_classifier import DEFAULT_MODEL_PATH, SpotClassifier

            self.classifier = SpotClassifier(model_path=model_path or DEFAULT_MODEL_PATH)

    def compute_occupancy(self, frame: np.ndarray) -> dict[str, bool]:
        if self.method == "yolo":
            return compute_occupancy(self.spots, self.detector.detect(frame), self.overlap_threshold)
        return self.classifier.compute_occupancy(frame, self.spots, self.probability_threshold)

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        occupancy = self.compute_occupancy(frame)
        annotated = _draw_overlay(frame, self.spots, occupancy)
        return PipelineResult(annotated_frame=annotated, occupancy=occupancy)
