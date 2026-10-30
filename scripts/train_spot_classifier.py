"""Train the per-spot empty/occupied CNN on PKLot-50 spot crops.

Images are split by image (never by crop) so test spots come from frames the
model never saw; see `is_test_image` in scripts/prepare_pklot.py.

Usage:
    python -m scripts.train_spot_classifier [--epochs 25] [--out models/spot_classifier.pt]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from scripts.prepare_pklot import IMAGES_DIR, OCCUPIED_CATEGORY, is_test_image, load_pklot, to_spots_json
from src.occupancy import Spot
from src.spot_classifier import DEFAULT_MODEL_PATH, SpotCNN, crop_spot, crops_to_tensor, pick_device


def build_crops(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    images, spots_by_image = load_pklot()
    crops, labels = [], []
    for image_id, image in images.items():
        if is_test_image(image_id) == train:
            continue
        frame = cv2.imread(str(IMAGES_DIR / image["file_name"]))
        annotations = spots_by_image[image_id]
        for entry, ann in zip(to_spots_json(annotations)["spots"], annotations):
            crops.append(crop_spot(frame, Spot.from_points(entry["id"], entry["points"])))
            labels.append(int(ann["category_id"] == OCCUPIED_CATEGORY))
    return crops_to_tensor(crops), torch.tensor(labels)


def augment(batch: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotations, flips and brightness/contrast jitter.

    Rotations matter: spots are seen at many orientations (portrait, landscape,
    straight-down views), not just the ones in PKLot.
    """
    out = []
    for img in batch:
        img = torch.rot90(img, random.randint(0, 3), dims=(1, 2))
        if random.random() < 0.5:
            img = torch.flip(img, dims=(2,))
        contrast = random.uniform(0.7, 1.3)
        brightness = random.uniform(-0.25, 0.25)
        out.append((img * contrast + brightness).clamp(-1, 1))
    return torch.stack(out)


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    preds = torch.cat([model(x[i : i + 512].to(device)).argmax(1).cpu() for i in range(0, len(x), 512)])
    return (preds == y).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = pick_device()

    x_train, y_train = build_crops(train=True)
    x_test, y_test = build_crops(train=False)
    print(f"Train crops: {len(y_train)} ({y_train.sum().item()} occupied)")
    print(f"Test crops:  {len(y_test)} ({y_test.sum().item()} occupied)  device: {device}")

    model = SpotCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Occupied spots are the minority class; weight them up so they aren't ignored.
    class_counts = torch.bincount(y_train, minlength=2).float()
    loss_fn = nn.CrossEntropyLoss(weight=(class_counts.sum() / (2 * class_counts)).to(device))

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y_train))
        total_loss = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            xb, yb = augment(x_train[idx]).to(device), y_train[idx].to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:2d}  loss {total_loss / len(order):.4f}  "
                f"train acc {accuracy(model, x_train, y_train, device):.3f}  "
                f"test acc {accuracy(model, x_test, y_test, device):.3f}"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.out)
    print(f"Saved {args.out} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
