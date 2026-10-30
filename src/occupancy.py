"""Overlap-based occupancy decisions for predefined parking spots."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon, box


@dataclass(frozen=True)
class Spot:
    """A single parking space, defined by a polygon in image pixel coordinates."""

    id: str
    polygon: Polygon

    @classmethod
    def from_points(cls, spot_id: str, points: list[tuple[float, float]]) -> "Spot":
        return cls(id=spot_id, polygon=Polygon(points))


def box_spot_overlap_ratio(vehicle_box: tuple[float, float, float, float], spot: Spot) -> float:
    """Fraction of the spot's area covered by the vehicle box (0..1)."""
    x1, y1, x2, y2 = vehicle_box
    vehicle_poly = box(x1, y1, x2, y2)
    if not spot.polygon.is_valid or spot.polygon.area == 0:
        return 0.0
    intersection = spot.polygon.intersection(vehicle_poly).area
    return intersection / spot.polygon.area


def is_occupied(
    spot: Spot,
    vehicle_boxes: list[tuple[float, float, float, float]],
    overlap_threshold: float = 0.3,
) -> bool:
    """A spot is occupied if any vehicle box covers at least `overlap_threshold` of it."""
    return any(
        box_spot_overlap_ratio(vehicle_box, spot) >= overlap_threshold
        for vehicle_box in vehicle_boxes
    )


def compute_occupancy(
    spots: list[Spot],
    vehicle_boxes: list[tuple[float, float, float, float]],
    overlap_threshold: float = 0.3,
) -> dict[str, bool]:
    """Returns {spot_id: is_occupied} for every spot."""
    return {
        spot.id: is_occupied(spot, vehicle_boxes, overlap_threshold)
        for spot in spots
    }
