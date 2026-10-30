"""Turn the small PKLot-50 subset into dashboard samples and evaluate the pipeline.

PKLot-50 (https://huggingface.co/datasets/ajankelo/pklot_50, CC-BY-4.0) is 50
PKLot images (640x640) with every parking space labelled `space-empty` or
`space-occupied` in COCO format. Download it into `data/pklot_50/`:

    data/pklot_50/annotations.json
    data/pklot_50/images/images/*.jpg   (unzipped from images.zip)

Usage:
    python -m scripts.prepare_pklot samples    # copy a few images + spot JSONs for the dashboard
    python -m scripts.prepare_pklot evaluate --method yolo|classifier
        # accuracy vs. ground truth on the 10 held-out test images
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PKLOT_DIR = Path("data/pklot_50")
ANNOTATIONS = PKLOT_DIR / "annotations.json"
IMAGES_DIR = PKLOT_DIR / "images" / "images"
SAMPLES_DIR = Path("data/samples")
SPOTS_DIR = Path("spots")

EMPTY_CATEGORY, OCCUPIED_CATEGORY = 1, 2

# A mix of both lots, crowded and sparse, sunny and overcast.
SAMPLE_IMAGES = {
    "pklot_pucpr_busy": "2012-10-29_13_33_14",
    "pklot_pucpr_sparse": "2012-09-16_12_23_10",
    "pklot_ufpr_half": "2013-04-16_07_20_01",
    "pklot_ufpr_mixed": "2012-12-16_18_50_16",
}


def is_test_image(image_id: int) -> bool:
    """Every 5th image is held out for testing (10 of 50, covering both lots)."""
    return image_id % 5 == 0


def load_pklot() -> tuple[dict[int, dict], dict[int, list[dict]]]:
    """Returns ({image_id: image_info}, {image_id: [spot annotations]})."""
    with open(ANNOTATIONS) as f:
        data = json.load(f)
    images = {image["id"]: image for image in data["images"]}
    spots_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for ann in data["annotations"]:
        if ann["category_id"] in (EMPTY_CATEGORY, OCCUPIED_CATEGORY):
            spots_by_image[ann["image_id"]].append(ann)
    return images, spots_by_image


def to_spots_json(annotations: list[dict]) -> dict:
    """COCO [x, y, w, h] boxes -> this project's {"spots": [{"id", "points"}]} format."""
    spots = []
    for i, ann in enumerate(annotations, start=1):
        x, y, w, h = ann["bbox"]
        points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        spots.append({"id": f"spot_{i}", "points": [[int(round(px)), int(round(py))] for px, py in points]})
    return {"spots": spots}


def find_image(images: dict[int, dict], prefix: str) -> dict:
    for image in images.values():
        if image["file_name"].startswith(prefix):
            return image
    raise KeyError(f"No PKLot-50 image starting with {prefix!r}")


def make_samples() -> None:
    images, spots_by_image = load_pklot()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    SPOTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, prefix in SAMPLE_IMAGES.items():
        image = find_image(images, prefix)
        annotations = spots_by_image[image["id"]]
        shutil.copy(IMAGES_DIR / image["file_name"], SAMPLES_DIR / f"{name}.jpg")
        with open(SPOTS_DIR / f"{name}.json", "w") as f:
            json.dump(to_spots_json(annotations), f, indent=2)
        occupied = sum(a["category_id"] == OCCUPIED_CATEGORY for a in annotations)
        print(f"{name}: {len(annotations)} spots, ground truth {occupied} occupied")


def evaluate(method: str, overlap_threshold: float, confidence: float, model_path: str) -> None:
    import cv2

    from src.occupancy import Spot
    from src.pipeline import ParkingPipeline

    images, spots_by_image = load_pklot()
    test_images = {image_id: image for image_id, image in images.items() if is_test_image(image_id)}
    pipeline = ParkingPipeline(
        [],
        method=method,
        model_path=model_path,
        overlap_threshold=overlap_threshold,
        confidence=confidence,
    )
    tp = fp = tn = fn = 0
    for image_id, image in test_images.items():
        annotations = spots_by_image[image_id]
        pipeline.spots = [
            Spot.from_points(entry["id"], entry["points"]) for entry in to_spots_json(annotations)["spots"]
        ]
        frame = cv2.imread(str(IMAGES_DIR / image["file_name"]))
        predicted = pipeline.process_frame(frame).occupancy
        for spot, ann in zip(pipeline.spots, annotations):
            actual = ann["category_id"] == OCCUPIED_CATEGORY
            if predicted[spot.id] and actual:
                tp += 1
            elif predicted[spot.id]:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"Method: {method}  test images: {len(test_images)}  spots: {total}")
    print(f"Accuracy:  {(tp + tn) / total:.3f}")
    print(f"Precision: {precision:.3f}  (occupied)")
    print(f"Recall:    {recall:.3f}  (occupied)")
    print(f"F1:        {f1:.3f}")
    print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["samples", "evaluate"])
    parser.add_argument("--method", choices=["yolo", "classifier"], default="yolo")
    parser.add_argument("--threshold", type=float, default=0.3, help="YOLO overlap threshold")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO detection confidence")
    parser.add_argument("--model", default=None, help="YOLO weights or classifier checkpoint")
    args = parser.parse_args()
    if args.command == "samples":
        make_samples()
    else:
        evaluate(args.method, args.threshold, args.confidence, args.model)


if __name__ == "__main__":
    main()
