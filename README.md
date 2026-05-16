
# Drone Human & Car Detection System
### Antlings Internship – AI/ML Technical Assessment

A computer vision pipeline for detecting and counting humans and cars in drone/aerial imagery, fine-tuned on the VisDrone dataset using YOLOv8.

---

## Table of Contents

- Overview
- Project Structure
- Dataset
- Setup & Installation
- Pipeline
  - [Task 01 – Dataset Understanding & Preprocessing](#task-01--dataset-understanding--preprocessing)
  - [Task 02 – Model Training](#task-02--model-training)
  - [Task 03 – Detection & Human Counting](#task-03--detection--human-counting)
  - [Task 04 – Object Tracking (Bonus)](#task-04--object-tracking-bonus)
  - [Task 05 – Evaluation & Visualization](#task-05--evaluation--visualization)
- Results
- Strengths & Limitations
- Demo Video

---

## Overview

This project builds an end-to-end aerial object detection pipeline that:

- Detects **humans** (pedestrians + people groups) and **cars** in drone images
- Displays annotated **bounding boxes** with class labels and confidence scores
- Outputs a **total human count** per image
- Optionally tracks objects across video frames using DeepSORT

The backbone is **YOLOv8n** fine-tuned on VisDrone with augmentation strategies tailored for small, densely packed aerial objects.

---

## Project Structure

```
├── configs/
│   ├── visdrone.yaml          # Dataset config (paths, class names)
│   └── train_config.yaml      # Full training hyperparameter config
├── src/
│   ├── dataset.py             # Task 01 – stats, visualizations, annotations
│   ├── train.py               # Task 02 – YOLOv8 training & validation
│   ├── detect.py              # Task 03 – inference, bbox drawing, counting
│   ├── track.py               # Task 04 – DeepSORT tracking
│   ├── evaluate.py            # Task 05 – mAP, FPS benchmarking
│   └── visualize.py           # Task 05 – charts, grids, summary cards
├── outputs/
│   ├── visualizations/        # Training curves, metrics card, count summary
│   ├── predictions/           # Annotated inference images
│   └── tracking/              # Tracking output frames/video
├── runs/
│   └── train/visdrone_yolov8n/
│       └── weights/
│           ├── best.pt        # Best checkpoint
│           └── last.pt        # Latest checkpoint
├── detection_pipeline_3.ipynb # Kaggle notebook (earlier iteration)
├── detection_pipeline_4.ipynb # Kaggle notebook (final version)
└── README.md
```

---

## Dataset

**VisDrone-DET 2019** — a large-scale benchmark of drone-captured imagery.

| Split      | Images  | Annotations |
|------------|---------|-------------|
| Train      | 6,471   | ~390,000    |
| Validation | 548     | ~33,000     |
| Test-Dev   | 1,610   | —           |

**Download:** [Kaggle – VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

The dataset contains 10 object classes. This pipeline targets three:

| YOLO Class ID | Original Label     | Mapped As  |
|---------------|--------------------|------------|
| 0             | pedestrian         | human      |
| 1             | people             | human      |
| 3             | car                | car        |

---

## Setup & Installation

> The pipeline was developed and trained on **Kaggle** with a dual T4 GPU environment. All paths in the notebooks use `/kaggle/` prefixes. To run locally, update the path variables at the top of each script/notebook.

### Dependencies

```bash
pip install ultralytics deep-sort-realtime opencv-python matplotlib seaborn pandas tqdm pyyaml
```

### Clone & Configure

```bash
git clone <repo-url>
cd <repo>
```

Update the `DATA_ROOT` path in `configs/visdrone.yaml` and `src/dataset.py` to point to your local VisDrone directory.

---

## Pipeline Walkthrough

### Task 01 – Dataset Understanding & Preprocessing

**Script:** `src/dataset.py`  
**Notebook cell:** *Task 01 — Dataset stats & preprocessing*

Parses all YOLO-format label files to build a per-annotation DataFrame, then generates:

- **Class distribution bar chart** — annotation counts for pedestrian, people, and car classes
- **Bounding box area histogram** — log-scaled area distributions showing the small-object challenge (median human bbox ≈ 200–400 px²)
- **Sample annotation grid** — 4 random training images with overlaid bounding boxes (green = human, blue = car)

**Key dataset observations:**

- **Severe class imbalance** — humans outnumber cars roughly 3:1; background dominates both
- **Small object density** — most human bboxes are under 400 px²; many are only a few pixels tall
- **Crowding** — images frequently contain 50–200 overlapping instances, complicating NMS
- **Viewpoint variation** — altitude and camera angle change significantly across scenes
- **Lighting & occlusion** — shadows, partial occlusions, and low-contrast backgrounds are common

**Augmentation strategy** (applied during training):

| Augmentation      | Value  | Rationale                                  |
|-------------------|--------|--------------------------------------------|
| Mosaic            | 1.0    | Simulates varied density and scale         |
| MixUp             | 0.15   | Improves generalisation on edge cases      |
| HSV jitter (S, V) | 0.7 / 0.4 | Handles lighting variation            |
| Horizontal flip   | 0.5    | Drone imagery has no preferred orientation |
| Scale             | 0.5    | Teaches robustness to altitude changes     |
| Translate         | 0.1    | Handles off-centre subjects                |

---

### Task 02 – Model Training

**Script:** `src/train.py`  
**Notebook cell:** *Task 02 — Training*

Fine-tunes **YOLOv8n** (nano variant, ~3.2M params) using the Ultralytics framework.

**Training configuration:**

| Parameter        | Value        |
|------------------|--------------|
| Base model       | `yolov8n.pt` |
| Epochs           | 50           |
| Image size       | 1280 × 1280  |
| Batch size       | 4 (per GPU)  |
| Devices          | 2× T4 GPU    |
| Optimizer        | AdamW        |
| Learning rate    | 0.001 → 0.01 (cosine LR) |
| Warmup epochs    | 3            |
| Confidence       | 0.25         |
| IoU threshold    | 0.45         |
| Max detections   | 300          |

The high image size (1280) is intentional — VisDrone's small targets are lost at standard 640px resolution.

Training checkpoints are saved every 5 epochs. The best weights (by `mAP@0.5`) are stored at `runs/train/visdrone_yolov8n/weights/best.pt`.

**Resuming training:**

```python
from ultralytics import YOLO
model = YOLO("runs/train/visdrone_yolov8n/weights/last.pt")
model.train(resume=True)
```

---

### Task 03 – Detection & Human Counting

**Script:** `src/detect.py`  
**Notebook cell:** *Task 03 — Detection & counting*

Runs inference on a directory of images using the trained `best.pt` checkpoint.

For each image the pipeline:
1. Runs YOLOv8 inference (conf ≥ 0.25, IoU ≤ 0.45, classes [0, 1, 3])
2. Draws colour-coded bounding boxes (green = human, blue = car) with confidence labels
3. Counts human detections (class 0 + class 1) and car detections (class 3)
4. Overlays the count on the image and saves the annotated result

**Counting logic:**

```python
human_count = sum(1 for cls in result.boxes.cls if cls in {0, 1})
car_count   = sum(1 for cls in result.boxes.cls if cls == 3)
```

Sample output per image:

```
humans=14  cars=3
```

---

### Task 04 – Object Tracking (Bonus)

**Script:** `src/track.py`  
**Notebook cell:** *Task 04 — Tracking*

Implements **DeepSORT** tracking on top of YOLOv8 detections for video input.

- Each confirmed track is assigned a persistent ID
- Track histories (deque of centroid positions) are drawn as motion trails
- Per-track class is stored so unique human and car tracks can be counted at the end

```python
from track import run_tracking

track_classes, track_history = run_tracking(
    weights  = "runs/train/visdrone_yolov8n/weights/best.pt",
    source   = "path/to/video.mp4",
    conf     = 0.30,
    iou      = 0.45,
    save_dir = "outputs/tracking",
    max_age  = 30,
)
print(f"Unique human tracks: {sum(1 for c in track_classes.values() if c == 0)}")
```

Tracking output (annotated video/frames) is saved to `outputs/tracking/`.

---

### Task 05 – Evaluation & Visualization

**Script:** `src/evaluate.py`, `src/visualize.py`  
**Notebook cell:** *Task 05 — Evaluation & visualization*

**Quantitative evaluation** uses the Ultralytics `.val()` API against the VisDrone validation split:

| Metric              | Description                             |
|---------------------|-----------------------------------------|
| `mAP@0.5`           | Standard detection metric               |
| `mAP@0.5:0.95`      | COCO-style strict metric                |
| Precision / Recall  | Per-class and mean                      |
| FPS                 | Inference speed benchmark (50 runs)     |

**Visualizations generated:**

- `training_curves.png` — box loss, cls loss, DFL loss, and mAP per epoch
- `metrics_card.png` — horizontal bar chart summarising all key metrics
- `prediction_grid.png` — 3×3 grid of annotated validation predictions
- `count_summary.png` — per-image human and car counts across the sample set

---

## Results



mAP@0.5 pedestrian: 0.439, people: 0.333, car: 0.772
Precision: 0.749, Recall: 0.558
FPS: 71.1, Latency: 14.3ms
50 epochs completed
YOLOv8n, batch=4, imgsz=1280, mosaic=1.0, mixup=0.15, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5, scale=0.5, translate=0.1
Sample prediction outputs and training curves are saved in `outputs/visualizations/`.

---

## Strengths & Limitations

**Strengths**

- High-resolution inference (1280px) preserves small object detail lost at 640px
- Mosaic + MixUp augmentation improves robustness to varying crowd density and scale
- Modular source layout — each task is an independent, importable module
- DeepSORT tracking provides persistent IDs without retraining

**Limitations**

- YOLOv8n is the smallest YOLO variant; larger variants (s/m/l) would improve mAP at the cost of speed
- Counting is frame-level, not scene-level — the same person is counted once per image
- Tracking requires video input; still-image datasets like VisDrone-DET do not natively support it
- At extreme crowd densities (100+ people per frame), overlapping proposals increase missed detections even after NMS tuning

**Challenges faced**

- VisDrone's small bounding boxes (median ≈ 300 px²) are below the effective detection threshold at 640px, making the 1280px image size mandatory
- Class imbalance between pedestrian/people vs. car required class-filtered training (classes=[0,1,3]) to prevent the model from over-fitting to the majority class
- Kaggle session resets required explicit weight backup steps between notebook saves

---

## Demo Video
https://drive.google.com/drive/folders/1HGIn9ID62G4WV83xA0ebpPXeseOxB8j9



The demo covers:
- Dataset exploration and sample annotations
- Training run summary and curves
- Live inference with bounding boxes and human count overlay
- (NOT COMPLETED) Tracking output with persistent IDs and motion trails
