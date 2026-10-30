from src.occupancy import Spot, box_spot_overlap_ratio, compute_occupancy, is_occupied

SQUARE_SPOT_POINTS = [(0, 0), (10, 0), (10, 10), (0, 10)]


def make_square_spot(spot_id: str = "spot_1") -> Spot:
    return Spot.from_points(spot_id, SQUARE_SPOT_POINTS)


def test_overlap_ratio_full_coverage():
    spot = make_square_spot()
    vehicle_box = (0, 0, 10, 10)
    assert box_spot_overlap_ratio(vehicle_box, spot) == 1.0


def test_overlap_ratio_no_overlap():
    spot = make_square_spot()
    vehicle_box = (100, 100, 110, 110)
    assert box_spot_overlap_ratio(vehicle_box, spot) == 0.0


def test_overlap_ratio_partial_coverage():
    spot = make_square_spot()
    vehicle_box = (5, 0, 15, 10)  # covers right half of the 10x10 spot
    assert box_spot_overlap_ratio(vehicle_box, spot) == 0.5


def test_is_occupied_true_above_threshold():
    spot = make_square_spot()
    vehicle_boxes = [(0, 0, 10, 10)]
    assert is_occupied(spot, vehicle_boxes, overlap_threshold=0.3) is True


def test_is_occupied_false_below_threshold():
    spot = make_square_spot()
    vehicle_boxes = [(9, 9, 20, 20)]  # tiny sliver of overlap
    assert is_occupied(spot, vehicle_boxes, overlap_threshold=0.3) is False


def test_is_occupied_false_with_no_vehicles():
    spot = make_square_spot()
    assert is_occupied(spot, [], overlap_threshold=0.3) is False


def test_compute_occupancy_multiple_spots():
    spots = [
        Spot.from_points("spot_1", SQUARE_SPOT_POINTS),
        Spot.from_points("spot_2", [(20, 0), (30, 0), (30, 10), (20, 10)]),
    ]
    vehicle_boxes = [(0, 0, 10, 10)]  # only covers spot_1

    occupancy = compute_occupancy(spots, vehicle_boxes, overlap_threshold=0.3)

    assert occupancy == {"spot_1": True, "spot_2": False}
