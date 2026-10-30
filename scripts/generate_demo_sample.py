"""Generate a small synthetic parking-lot image + matching spot definitions.

This is only a stand-in so the dashboard works out of the box without
internet access to a real dataset. For real evaluation, download PKLot
(https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz, mirrored on Kaggle as
"andrewmvd/pklot-dataset") and point the dashboard at its images instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

OUTPUT_IMAGE = Path("data/samples/demo_lot.jpg")
OUTPUT_SPOTS = Path("spots/demo_lot.json")

ROWS, COLS = 2, 5
SPOT_W, SPOT_H = 120, 220
MARGIN_X, MARGIN_Y = 40, 60
GAP_X, GAP_Y = 20, 40

# (row, col) pairs that should be drawn as "occupied" in the demo image.
OCCUPIED_CELLS = {(0, 1), (0, 3), (1, 0), (1, 4)}


def _spot_rect(row: int, col: int) -> tuple[int, int, int, int]:
    x1 = MARGIN_X + col * (SPOT_W + GAP_X)
    y1 = MARGIN_Y + row * (SPOT_H + GAP_Y)
    return x1, y1, x1 + SPOT_W, y1 + SPOT_H


def generate() -> None:
    width = MARGIN_X * 2 + COLS * SPOT_W + (COLS - 1) * GAP_X
    height = MARGIN_Y * 2 + ROWS * SPOT_H + (ROWS - 1) * GAP_Y
    frame = np.full((height, width, 3), (90, 90, 90), dtype=np.uint8)

    spots_data = []
    for row in range(ROWS):
        for col in range(COLS):
            x1, y1, x2, y2 = _spot_rect(row, col)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            spot_id = f"spot_{row}_{col}"
            spots_data.append(
                {"id": spot_id, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}
            )

            if (row, col) in OCCUPIED_CELLS:
                pad = 12
                cv2.rectangle(
                    frame,
                    (x1 + pad, y1 + pad),
                    (x2 - pad, y2 - pad),
                    (40, 40, 180),
                    -1,
                )

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SPOTS.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), frame)
    with open(OUTPUT_SPOTS, "w") as f:
        json.dump({"spots": spots_data}, f, indent=2)

    print(f"Wrote {OUTPUT_IMAGE} and {OUTPUT_SPOTS}")


if __name__ == "__main__":
    generate()
