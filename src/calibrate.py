"""Interactive tool to define parking-spot polygons on a reference frame.

Usage:
    python -m src.calibrate <image_path> <output_spots_json>

Controls:
    left click   - add a point to the current spot polygon
    n            - finish current spot (needs >= 3 points) and start a new one
    u            - undo the last point
    s            - save all finished spots to the output JSON and quit
    q / Esc      - quit without saving
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

WINDOW_NAME = "Calibrate parking spots"


def _draw(frame, finished_spots: list[list[tuple[int, int]]], current_points: list[tuple[int, int]]):
    display = frame.copy()
    for spot_points in finished_spots:
        cv2.polylines(display, [_as_array(spot_points)], isClosed=True, color=(0, 255, 0), thickness=2)
    if current_points:
        cv2.polylines(display, [_as_array(current_points)], isClosed=False, color=(0, 165, 255), thickness=2)
        for point in current_points:
            cv2.circle(display, point, 4, (0, 165, 255), -1)
    cv2.putText(
        display,
        f"spots: {len(finished_spots)}  points in progress: {len(current_points)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.imshow(WINDOW_NAME, display)


def _as_array(points: list[tuple[int, int]]):
    import numpy as np

    return np.array(points, dtype=int)


def run_calibration(image_path: str, output_path: str) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    finished_spots: list[list[tuple[int, int]]] = []
    current_points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            _draw(frame, finished_spots, current_points)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    _draw(frame, finished_spots, current_points)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("Quit without saving.")
            return
        if key == ord("u") and current_points:
            current_points.pop()
            _draw(frame, finished_spots, current_points)
        elif key == ord("n"):
            if len(current_points) >= 3:
                finished_spots.append(current_points.copy())
                current_points.clear()
                _draw(frame, finished_spots, current_points)
            else:
                print("Need at least 3 points to finish a spot.")
        elif key == ord("s"):
            if len(current_points) >= 3:
                finished_spots.append(current_points.copy())
                current_points.clear()
            break

    cv2.destroyAllWindows()

    if not finished_spots:
        print("No spots defined, nothing saved.")
        return

    spots_data = [
        {"id": f"spot_{i + 1}", "points": [[int(x), int(y)] for x, y in points]}
        for i, points in enumerate(finished_spots)
    ]
    with open(output_path, "w") as f:
        json.dump({"spots": spots_data}, f, indent=2)
    print(f"Saved {len(spots_data)} spots to {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Reference frame to define parking spots on")
    parser.add_argument("output_path", help="Where to save the spots JSON file")
    args = parser.parse_args(argv)
    run_calibration(args.image_path, args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
