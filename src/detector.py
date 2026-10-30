"""Vehicle detection wrapper around a pretrained Ultralytics YOLO model."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}


class VehicleDetector:
    """Loads a YOLO model once and returns vehicle bounding boxes per frame."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self._vehicle_class_ids = {
            class_id
            for class_id, name in self.model.names.items()
            if name in VEHICLE_CLASS_NAMES
        }

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Returns a list of (x1, y1, x2, y2) boxes for detected vehicles in `frame`."""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        boxes: list[tuple[float, float, float, float]] = []
        for result in results:
            for box_data in result.boxes:
                class_id = int(box_data.cls[0])
                if class_id not in self._vehicle_class_ids:
                    continue
                x1, y1, x2, y2 = box_data.xyxy[0].tolist()
                boxes.append((x1, y1, x2, y2))
        return boxes
