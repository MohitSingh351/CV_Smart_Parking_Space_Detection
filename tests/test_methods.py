import cv2
import numpy as np
import torch

from src.occupancy import Spot
from src.spot_classifier import CROP_SIZE, SpotCNN, crop_spot, crops_to_tensor

EMPTY_SPOT = Spot.from_points("empty", [(10, 10), (60, 10), (60, 110), (10, 110)])
CAR_SPOT = Spot.from_points("car", [(80, 10), (130, 10), (130, 110), (80, 110)])


def make_frame() -> np.ndarray:
    """Flat grey 'asphalt' with a white "car" with dark windows in CAR_SPOT."""
    frame = np.full((120, 140, 3), 100, dtype=np.uint8)
    cv2.rectangle(frame, (88, 20), (122, 100), (230, 230, 230), -1)
    for y in (30, 70):
        cv2.rectangle(frame, (93, y), (117, y + 15), (20, 20, 20), -1)
    return frame


def test_crop_spot_has_fixed_size_and_handles_out_of_frame_spots():
    frame = make_frame()
    assert crop_spot(frame, CAR_SPOT).shape == (CROP_SIZE, CROP_SIZE, 3)
    outside = Spot.from_points("outside", [(500, 500), (600, 500), (600, 600)])
    assert crop_spot(frame, outside).shape == (CROP_SIZE, CROP_SIZE, 3)


def test_spot_cnn_outputs_two_logits_per_crop():
    frame = make_frame()
    batch = crops_to_tensor([crop_spot(frame, EMPTY_SPOT), crop_spot(frame, CAR_SPOT)])
    assert batch.shape == (2, 3, CROP_SIZE, CROP_SIZE)
    with torch.no_grad():
        assert SpotCNN().eval()(batch).shape == (2, 2)
