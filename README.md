# Smart Parking Space Detection

Look at a picture (or video) from a parking-lot camera and answer, for every
parking space: **is it occupied or free?** The app colours each space
**red (occupied)** or **green (free)** and shows the counts, e.g.
`Occupied: 9/22  Free: 13/22`.

---

## Table of contents

1. [The problem in one minute](#1-the-problem-in-one-minute)
2. [How the system works (big picture)](#2-how-the-system-works-big-picture)
3. [Step 1 — Marking the parking spots (calibration)](#3-step-1--marking-the-parking-spots-calibration)
4. [Step 2 — Deciding occupied or free (two methods)](#4-step-2--deciding-occupied-or-free-two-methods)
5. [How the CNN was trained](#5-how-the-cnn-was-trained)
6. [Results](#6-results)
7. [Thresholds — and why you don't tune them per image](#7-thresholds--and-why-you-dont-tune-them-per-image)
8. [How to run it](#8-how-to-run-it)
9. [Using your own image or video](#9-using-your-own-image-or-video)
10. [Project structure (what each file does)](#10-project-structure-what-each-file-does)
11. [Limitations and future work](#11-limitations-and-future-work)
12. [Credits](#12-credits)

---

## 1. The problem in one minute

Finding a free parking space wastes time and fuel. Many parking lots already
have a fixed security camera. If a computer can look at that camera image and
tell which spaces are free, a sign or app can guide drivers straight to one.

The camera never moves, so the **positions of the parking spaces in the image
never change**. That makes the problem much simpler: we only need to decide,
for each known space, whether a car is in it.

---

## 2. How the system works (big picture)

```
 ┌───────────────┐   once per camera    ┌──────────────────────┐
 │ camera image  │ ───────────────────▶ │ spots/<name>.json    │  (polygons of every
 └───────┬───────┘   (calibration)      │  parking space)      │   parking space)
         │                              └──────────┬───────────┘
         │ every image / video frame               │
         ▼                                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Occupancy decision — choose one method:                    │
 │   A) Spot classifier (CNN):  crop each space → "occupied?"  │
 │   B) YOLO + overlap:         find cars → do they cover a    │
 │                              space?                         │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Draw result: red = occupied, green = free, + counts        │
 │  (Streamlit dashboard in the browser)                       │
 └─────────────────────────────────────────────────────────────┘
```

So there are two stages:

1. **Calibration (done once):** draw the outline of every parking space.
2. **Detection (done for every image/frame):** decide occupied/free for each
   outline, then draw the coloured result.

---

## 3. Step 1 — Marking the parking spots (calibration)

Each parking space is stored as a **polygon** — a list of corner points in
pixel coordinates. Four corners is typical, which also handles slanted or
perspective-distorted spaces.

Example (`spots/demo_lot.json`, shortened):

```json
{
  "spots": [
    {"id": "spot_0_0", "points": [[40, 60], [160, 60], [160, 280], [40, 280]]},
    {"id": "spot_0_1", "points": [[180, 60], [300, 60], [300, 280], [180, 280]]}
  ]
}
```

Some sample files also contain `"ground_truth": "occupied"` / `"empty"` for
each spot — the correct answer, used to measure accuracy. The app ignores it.

**How you create it:** run the click tool on a frame from your camera
(details in [section 9](#9-using-your-own-image-or-video)) and click the
corners of each space.

**Why not detect the spaces automatically?** A pretrained detector such as
YOLO only knows 80 everyday object types ("car", "person", "cell phone", …).
"Parking space" is not one of them. Since the camera is fixed, drawing the
spaces once by hand is the standard, reliable approach.

---

## 4. Step 2 — Deciding occupied or free (two methods)

The project compares **two different ideas**. You pick one in the app's
sidebar.

### Method A — Spot classifier (CNN) — *"Is this space occupied?"*

Instead of searching the whole image for cars, look at **one parking space at
a time** and ask a yes/no question.

```
full image ──▶ cut out space #1 ──▶ resize to 64×64 ──▶ CNN ──▶ P(occupied) = 0.98 → OCCUPIED
           ──▶ cut out space #2 ──▶ resize to 64×64 ──▶ CNN ──▶ P(occupied) = 0.02 → FREE
           ...
```

1. **Crop:** cut out the rectangle around the space's polygon.
2. **Resize:** every crop becomes 64×64 pixels, so small far-away spaces and
   big close-up spaces look the same size to the network.
3. **Classify:** a small Convolutional Neural Network (CNN) outputs the
   probability that the space is occupied.
4. **Decide:** probability ≥ 0.5 → occupied.

**The CNN (`src/spot_classifier.py`)** — small on purpose (~0.1 million
parameters, file size 323 KB), so it trains in under a minute and runs fast:

| Layer block | What it does | Output size |
|---|---|---|
| Input | 64×64 colour crop | 3 × 64 × 64 |
| Conv 3×3 → BatchNorm → ReLU → MaxPool | finds simple patterns (edges, colour changes) | 16 × 32 × 32 |
| Conv 3×3 → BatchNorm → ReLU → MaxPool | combines them (corners, shapes) | 32 × 16 × 16 |
| Conv 3×3 → BatchNorm → ReLU → MaxPool | larger parts (windows, wheels, car outline) | 64 × 8 × 8 |
| Conv 3×3 → BatchNorm → ReLU → MaxPool | whole-object features | 96 × 4 × 4 |
| Global average pool → Dropout → Linear | final decision | 2 scores: [empty, occupied] |

**Why it's a good idea:** it never has to *name* an object. It doesn't matter
whether a car looks like a "car" or a "phone" from above — only whether the
space looks empty or not. It also judges each space on its own pixels, so a
car can't "spill" into the neighbouring space.

### Method B — YOLO + overlap — *"Is this object a car?"*

Use a famous pretrained object detector, **YOLOv8-nano**, to find vehicles,
then check which spaces they cover.

```
full image ──▶ YOLO ──▶ boxes labelled car / truck / bus / motorcycle
                         │
   for every space:  what fraction of the space is covered by a car box?
                         │
                  ≥ 30 % → OCCUPIED,  otherwise FREE
```

1. **Detect:** YOLO returns boxes with a label and a confidence score (0–1).
   Only `car`, `truck`, `bus`, `motorcycle` with confidence ≥ 0.35 are kept.
2. **Overlap:** for each space, compute *(area of space covered by a car box)
   ÷ (area of space)* using polygon geometry (the `shapely` library).
3. **Decide:** ≥ 0.30 → occupied.

**Strengths:** no training needed, works on clear photos taken at a normal
angle. **Weakness:** YOLO was trained on the COCO dataset — mostly street-level
photos where cars are seen from the side. From high above, it is unsure about
cars, and from straight above it even calls them **"cell phone"** or
**"parking meter"** (a car roof with a dark windscreen looks like a phone!).
Those labels are not vehicle classes, so those cars are ignored.

### Comparison at a glance

| | Spot classifier (CNN) | YOLO + overlap |
|---|---|---|
| Question asked | "Is this space occupied?" | "Where are the cars?" |
| Needs training on parking data | Yes (done, ~45 s) | No (uses pretrained COCO model) |
| Top-down / high views | Very good | Poor (misses or mislabels cars) |
| Clear, angled views of new lots | Mixed | Good |
| Car overlapping two spaces in the image | Judges each space separately | Its box can mark both |
| Speed | Very fast (tiny network on small crops) | Slower (big network on full image) |

---

## 5. How the CNN was trained

**Data — PKLot-50.** 50 real images from the PKLot research dataset (two
parking lots in Brazil, different days and weather), with **every space
labelled** empty or occupied: 7,250 labelled spaces in total.

**Fair train/test split.** Every 5th image (10 images, 1,368 spaces) is kept
aside as the **test set** and never used for training. The split is by
*image*, not by space, so the test spaces really come from photos the model
has never seen.

**Training recipe** (`scripts/train_spot_classifier.py`):

- 5,882 training crops (1,864 occupied).
- **Data augmentation** — each crop is randomly rotated by 0/90/180/270°,
  flipped, and given random brightness/contrast. This teaches the model that a
  car is still a car when seen from another angle or in different light, and
  is why it also works on straight-down images it never trained on.
- **Class weights** — there are fewer occupied than empty examples, so mistakes
  on occupied spaces are weighted more, so the model doesn't just learn
  "always empty".
- Optimiser AdamW, 25 epochs, cosine learning-rate schedule, Apple GPU (MPS).
  Training takes **~45 seconds**.

Training log (test accuracy climbs to ~97 %):

```
epoch  1  loss 0.2135  train acc 0.939  test acc 0.906
epoch 10  loss 0.0847  train acc 0.977  test acc 0.971
epoch 25  loss 0.0436  train acc 0.988  test acc 0.974
```

---

## 6. Results

### How accuracy is measured (simple explanation)

Each space ends up in one of four boxes:

| | Really occupied | Really empty |
|---|---|---|
| **Predicted occupied** | ✅ True Positive (TP) | ❌ False Positive (FP) — "false alarm" |
| **Predicted empty** | ❌ False Negative (FN) — "missed car" | ✅ True Negative (TN) |

- **Accuracy** = how many spaces were right overall.
- **Precision** = when it says "occupied", how often is it right? (few false alarms)
- **Recall** = of all real cars, how many did it find? (few missed cars)
- **F1** = one number combining precision and recall.

Accuracy alone can mislead: if most spaces are empty, a system that always
says "empty" already scores high. That's why recall and F1 matter.

### Main result — PKLot test set (10 held-out images, 1,368 spaces)

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Spot classifier (CNN)** | **0.974** | **0.979** | **0.966** | **0.973** |
| YOLOv8n + overlap (confidence 0.10) | 0.645 | 0.994 | 0.243 | 0.391 |
| YOLOv8n + overlap (confidence 0.35, default) | 0.594 | 0.989 | 0.134 | 0.236 |

**Reading it:** YOLO is almost never wrong when it says "occupied"
(precision ≈ 0.99), but it **misses most cars** (recall 13–24 %) because PKLot
is filmed from high above. The CNN finds 96.6 % of cars with very few false
alarms.

Reproduce with:

```bash
python -m scripts.prepare_pklot evaluate --method classifier
python -m scripts.prepare_pklot evaluate --method yolo
```

### Generalisation — new images from other parking lots

These images come from parking lots **neither method was trained on**. All
runs use the **same fixed default settings** (no per-image tuning). Numbers
are spaces judged correctly.

| Image (`data/samples/`) | View | Spaces | Spot classifier | YOLO |
|---|---|---|---|---|
| `simple_lot.jpg` | PKLot crop, high & blurry | 12 | **12/12** | 8/12 |
| `topdown_lot.jpg` | straight down | 14 | **14/14** | 4/14 |
| `angled_lot.jpg` | straight down, slanted bays | 10 | **9/10** | 5/10 |
| `cctv_lot.jpg` | CCTV, angled | 22 | **21/22** | 18/22 |
| `drone_lot.jpg` | drone, angled | 17 | 14/17 | **17/17** |
| `apartment_lot.jpg` | high building, angled | 48 | 15/48 | **40/48** |

**What this shows:**

- **Top-down and high views:** the CNN wins clearly. YOLO cannot recognise
  cars from straight above (in `topdown_lot` it finds **zero** cars).
- **Clear angled photos of unfamiliar lots:** YOLO can win. On
  `apartment_lot` the CNN marks almost everything occupied — its training data
  was only 2 lots, and this lot's dark, stained concrete and thick slanted
  white lines look "busy" to it, like a car.
- **Typical failure cases** for both: cars hidden by trees (`cctv_lot` spot 8),
  bushes inside a space (`drone_lot` spots 12–13), a car driving past
  (`cctv_lot`, YOLO).

**Conclusion:** on the proper benchmark the spot classifier is clearly better
(97.4 % vs 59–65 %). On completely new lots, each method has strengths; the
CNN's weakness is limited training variety, which more data would fix (see
[section 11](#11-limitations-and-future-work)).

---

## 7. Thresholds — and why you don't tune them per image

The sidebar has sliders:

| Method | Slider | Default | Meaning |
|---|---|---|---|
| CNN | Occupied probability threshold | 0.5 | probability ≥ this → occupied |
| YOLO | Vehicle detection confidence | 0.35 | ignore YOLO boxes less sure than this |
| YOLO | Occupied overlap threshold | 0.30 | a car box must cover ≥ 30 % of the space |

Lowering YOLO's confidence finds more cars (higher recall) but also more junk
boxes (lower precision). That is the classic **precision–recall trade-off**.

**In real life you don't change sliders for every image.** A parking camera
is fixed, so its view never changes. Thresholds are set **once per camera**,
on a few labelled frames from that camera (pick the value with the best F1),
and then frozen. Every new frame from that camera then uses the same settings
automatically.

Also, needing to tune at all is a sign the model is seeing unfamiliar data.
On images similar to its training data, the CNN's probabilities are close to
**1.0 for cars and 0.0 for empty spaces**, so any threshold between 0.1 and
0.9 gives the same answer. The real fix for unfamiliar lots is better training
data, not more knob-turning.

All results in this README use the fixed defaults above.

---

## 8. How to run it

### Install (once)

Needs Python 3.12 (PyTorch may not support the newest Python yet).

```bash
git clone https://github.com/MohitSingh351/CV_Smart_Parking_Space_Detection.git
cd CV_Smart_Parking_Space_Detection
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The YOLO method needs `yolov8n.pt` (6 MB). It downloads automatically on first
use; on a slow connection download it yourself (the command can resume if
interrupted):

```bash
curl -L -C - --retry 10 -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
```

The trained CNN (`models/spot_classifier.pt`) is already included.

### Start the dashboard

```bash
streamlit run app.py
```

It opens in your browser at <http://localhost:8501>. Then:

1. **Source type:** Image or Video.
2. **Sample image/video:** pick a file from `data/samples/`.
3. **Parking-spot definition:** filled in automatically with the file of the
   same name (`cctv_lot.jpg` → `spots/cctv_lot.json`). A warning appears if
   they don't match.
4. **Detection method:** Spot classifier (default) or YOLO.
5. Click **Run detection** (or **Run on video**).

> If you edit any code while Streamlit is running, stop it (Ctrl+C) and start
> it again so it reloads everything.

---

## 9. Using your own image or video

### Your own image

1. **Take/choose a photo** from a fixed, elevated position (window, rooftop,
   CCTV). Resize very large photos to about 1280 px wide first.
2. **Copy it** into the samples folder:
   ```bash
   cp ~/Downloads/my_lot.jpg data/samples/my_lot.jpg
   ```
3. **Draw the spaces once** — the output file must have the **same name** as
   the image:
   ```bash
   python -m src.calibrate data/samples/my_lot.jpg spots/my_lot.json
   ```
   A window opens:

   | Key / mouse | Action |
   |---|---|
   | Left-click | add a corner to the current space |
   | `n` | finish this space, start the next one |
   | `u` | undo the last corner |
   | `s` | save and quit |
   | `q` / Esc | quit without saving |

4. **Run the dashboard** and select `my_lot.jpg`.

### Your own video

Video is just many frames from the same fixed camera, so you draw the spaces
once on one frame.

1. Use a **fixed camera** (tripod or CCTV), a **short** clip (10–60 s), at most
   ~1280 px wide. Supported: `.mp4`, `.avi`, `.mov`.
2. Copy it: `cp ~/Downloads/my_lot.mp4 data/samples/my_lot.mp4`
3. Save one frame **outside** `data/samples/` and draw on it:
   ```bash
   python -c "import cv2; ok,f=cv2.VideoCapture('data/samples/my_lot.mp4').read(); cv2.imwrite('my_lot_frame.png',f)"
   python -m src.calibrate my_lot_frame.png spots/my_lot.json
   ```
4. In the app choose **Source type: Video**, pick `my_lot.mp4`, click
   **Run on video**.

**Important:** spaces must be drawn on an image with the **same resolution and
camera position** as the one you analyse. If the camera moves or the image is
resized, draw the spaces again.

### Retrain or re-evaluate the CNN (optional)

Download [PKLot-50](https://huggingface.co/datasets/ajankelo/pklot_50)
(`annotations.json` + `images.zip`, ~5 MB) into `data/pklot_50/` and unzip the
images to `data/pklot_50/images/images/`. Then:

```bash
python -m scripts.prepare_pklot samples                       # sample images + spot files
python -m scripts.train_spot_classifier                       # train (~1 min)
python -m scripts.prepare_pklot evaluate --method classifier  # or --method yolo
```

### Run the tests

```bash
pytest tests/
```

---

## 10. Project structure (what each file does)

```
app.py                       Streamlit dashboard (the web page you use)
src/
  calibrate.py               click tool to draw parking spaces → spots/*.json
  pipeline.py                loads the spaces, runs the chosen method, draws red/green overlay
  spot_classifier.py         Method A: crop each space + the small CNN
  detector.py                Method B: YOLOv8 vehicle detector (keeps car/truck/bus/motorcycle)
  occupancy.py               Method B: polygon overlap maths (how much of a space a car covers)
models/
  spot_classifier.pt         trained CNN weights (323 KB)
spots/                       one JSON file of space polygons per camera view
data/
  samples/                   images/videos shown in the app
  pklot_50/                  PKLot-50 training data (downloaded separately)
scripts/
  train_spot_classifier.py   trains the CNN on PKLot-50
  prepare_pklot.py           makes sample images from PKLot-50; evaluates either method
  generate_demo_sample.py    makes a synthetic test image (demo_lot.jpg)
tests/
  test_occupancy.py          tests for the overlap maths
  test_methods.py            tests for cropping and the CNN's input/output shapes
requirements.txt             Python packages to install
```

About the sample images: the `pklot_*.jpg` images and `simple_lot.jpg` (a 2×
crop of a test-set PKLot image) come from PKLot-50. `demo_lot.jpg` is
synthetic — its "cars" are plain coloured rectangles, useful only to check
that the app runs; no method will see cars in it.

---

## 11. Limitations and future work

**Limitations**

- **Small training set:** the CNN learned from 50 images of only 2 parking
  lots, so it can struggle on very different-looking lots (see
  `apartment_lot`).
- **Pretrained YOLO doesn't know overhead cars:** it was trained on
  street-level photos.
- **Occlusion:** cars hidden behind trees, or bushes inside a space, confuse
  both methods.
- **Rectangular crops:** for strongly slanted spaces, the crop around the
  polygon also includes parts of the neighbouring spaces.
- **Video:** each frame is judged on its own, so a borderline space can
  flicker, and a car driving past can briefly mark spaces as occupied.
- **Manual calibration:** the spaces must be drawn once per camera.

**Future work**

- Train the CNN on the full PKLot / CNRPark-EXT datasets (hundreds of
  thousands of labelled spaces from many lots and cameras).
- Fine-tune YOLO on aerial parking data (e.g. CARPK) so it recognises cars
  from above; try a larger model and larger input size.
- Straighten each slanted space with a perspective warp before classifying.
- Video: smooth decisions over several frames (majority vote), skip frames
  for speed, save the annotated video.
- Per-camera automatic threshold selection from a few labelled frames.
- Occupancy history, "lot full" alerts, and guidance to the nearest free space.

---

## 12. Credits

- **PKLot** — Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A.,
  Koerich, A. *PKLot – A robust dataset for parking lot classification.*
  Expert Systems with Applications, 2015.
- **PKLot-50** subset — <https://huggingface.co/datasets/ajankelo/pklot_50>
  (CC-BY-4.0).
- **YOLOv8** — Ultralytics, <https://github.com/ultralytics/ultralytics>.
- Built with PyTorch, OpenCV, Shapely and Streamlit.
