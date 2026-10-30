"""Per-spot CNN classifier: crop each parking space and ask "empty or occupied?".

Instead of detecting and naming vehicles in the whole frame, every spot is
cropped, resized to a fixed size and classified on its own. This is the
approach used in the PKLot / CNRPark literature (e.g. mAlexNet).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from src.occupancy import Spot

CROP_SIZE = 64
DEFAULT_MODEL_PATH = "models/spot_classifier.pt"


def crop_spot(frame: np.ndarray, spot: Spot, size: int = CROP_SIZE) -> np.ndarray:
    """Bounding-box crop of the spot polygon, resized to (size, size), BGR uint8."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = spot.polygon.bounds
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(np.ceil(x2)), width), min(int(np.ceil(y2)), height)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def crops_to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    """BGR uint8 crops -> normalized float tensor of shape (N, 3, H, W) in RGB."""
    batch = np.stack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]).astype(np.float32)
    batch = (batch / 255.0 - 0.5) / 0.5
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


class SpotCNN(nn.Module):
    """Small CNN (4 conv blocks, ~0.1M params); output logits for [empty, occupied]."""

    def __init__(self) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SpotClassifier:
    """Loads a trained SpotCNN and returns occupancy decisions for a frame."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: torch.device | None = None):
        self.device = device or pick_device()
        self.model = SpotCNN()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def occupied_probabilities(self, frame: np.ndarray, spots: list[Spot]) -> dict[str, float]:
        if not spots:
            return {}
        batch = crops_to_tensor([crop_spot(frame, spot) for spot in spots]).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1)[:, 1].cpu().tolist()
        return {spot.id: prob for spot, prob in zip(spots, probs)}

    def compute_occupancy(
        self, frame: np.ndarray, spots: list[Spot], probability_threshold: float = 0.5
    ) -> dict[str, bool]:
        probs = self.occupied_probabilities(frame, spots)
        return {spot_id: prob >= probability_threshold for spot_id, prob in probs.items()}
