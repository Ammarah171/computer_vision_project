# Drone Human & Car Detection System
### Antlings Internship – AI/ML Technical Assessment

A computer vision pipeline for detecting and counting humans and cars in drone/aerial imagery, fine-tuned on the VisDrone dataset using YOLOv8n.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [Pipeline](#pipeline)
  - [Task 01 – Dataset Understanding & Preprocessing](#task-01--dataset-understanding--preprocessing)
  - [Task 02 – Model Training](#task-02--model-training)
  - [Task 03 – Detection & Human Counting](#task-03--detection--human-counting)
  - [Task 04 – Object Tracking (Bonus)](#task-04--object-tracking-bonus)
  - [Task 05 – Evaluation & Visualization](#task-05--evaluation--visualization)
- [Results](#results)
- [Strengths & Limitations](#strengths--limitations)
- [Demo Video](#demo-video)

---

## Overview

This project builds an end-to-end aerial object detection pipeline that:

- Detects **humans** (pedestrians + people groups) and **cars** in drone images
- Displays annotated **bounding boxes** with class labels and confidence scores
- Outputs a **total human and car count** per image via an on-frame overlay panel
- Optionally tracks objects across video frames using **DeepSORT** with persistent IDs and motion trails

The backbone is **YOLOv8n** fine-tuned on VisDrone with augmentation strategies tailored for small, densely packed aerial objects.

**Key results at a glance:**

| Metric | Value |
|---|---|
| Pedestrian mAP@0.5 | 0.439 |
| People mAP@0.5 | 0.333 |
| Car mAP@0.5 | 0.772 |
| Precision | 0.749 |
| Recall | 0.558 |
| Mean FPS | 71.1 |
| Inference latency | 14.3 ms |

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
│           ├── best.pt        # Best checkpoint (highest val mAP)
│           └── last.pt        # Latest checkpoint
└── README.md
```

---

## Dataset

**VisDrone-DET 2019** — a large-scale benchmark of drone-captured imagery collected by the AISKYEYE team across 14 cities in China at varying altitudes, weather conditions, and lighting.

| Split | Images | Annotations |
|---|---|---|
| Train | 6,471 | ~390,000 |
| Validation | 548 | ~33,000 |
| Test-Dev | 1,610 | — |
| Test-Challenge | 1,580 | — |

**Download:** [Kaggle – VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

The dataset contains 10 object classes. This pipeline targets three:

| YOLO Class ID | Original Label | Mapped As |
|---|---|---|
| 0 | pedestrian | human |
| 1 | people | human |
| 3 | car | car |

Classes 0 and 1 are both counted as humans because from a detection and counting perspective, a person in a crowd is still a person. The `pedestrian`/`people` distinction in VisDrone is also subjective and inconsistent between annotators.

---

## Setup & Installation

> The pipeline was developed and trained on **Kaggle** with a T4 GPU environment. All paths in the notebooks use `/kaggle/` prefixes. To run locally, update the path variables in `configs/visdrone.yaml` and the source scripts.

### Dependencies

```bash
pip install ultralytics deep-sort-realtime opencv-python matplotlib seaborn pandas tqdm pyyaml
```

### Clone & Configure

```bash
git clone <repo-url>
cd <repo>
```

Update `path` in `configs/visdrone.yaml` to point to your local VisDrone directory:

```yaml
path: /your/local/path/VisDrone_Dataset
train: VisDrone2019-DET-train/images
val:   VisDrone2019-DET-val/images
test:  VisDrone2019-DET-test-dev/images
nc:    10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

---

## Pipeline

### Task 01 – Dataset Understanding & Preprocessing

**Script:** `src/dataset.py`

Parses all YOLO-format label files to build a per-annotation DataFrame and generates three visualizations:

- **Class distribution bar chart** — annotation counts for pedestrian, people, and car
- **Bounding box area histogram** — log-scaled area distributions showing the small-object challenge
- **Sample annotation grid** — 4 random training images with ground truth boxes overlaid (green = human, blue = car)

#### Dataset Statistics

**Annotation counts (train split, target classes):**

| Class | Count |
|---|---|
| car | 144,867 |
| pedestrian | 79,337 |
| people | 27,059 |

**Human bounding box size statistics:**

| Stat | Value |
|---|---|
| Min area | 3 px² |
| 25th percentile | 135 px² |
| Median area | 304 px² |
| 75th percentile | 704 px² |
| Max area | 72,670 px² |

#### Challenges

**1. Tiny Object Size**
A human with a median area of 304 px² in a 1920×1080 image occupies roughly 17×18 pixels. At the default imgsz=640, YOLOv8 downscales the image 3×, making that human only ~34 px² — essentially invisible to the network. This is why `imgsz=1280` is mandatory for this dataset.

**2. Extreme Scale Variation**
Human bounding box areas range from 3 px² to 72,670 px² — a 24,000× difference — because drones fly at different altitudes. The model must detect humans across all these scales simultaneously.

**3. Severe Class Imbalance**
Cars (144,867) significantly outnumber humans (106,396 combined). Non-target classes like motor (29,647) and van (24,956) are also visually similar to target classes, creating confusion during training.

**4. Dense Crowds**
With approximately 53 objects per image on average, many scenes contain hundreds of overlapping bounding boxes. NMS struggles to separate individual detections in tightly packed groups.

**5. Annotation Ambiguity**
The `pedestrian` vs `people` distinction is subjective — the same person could be labeled either way depending on the annotator. This inconsistency reduces model confidence on human classes.

**6. Long Tail Distribution**
Most objects are tiny (75th percentile = 704 px²) but a few are very large (max = 72,670 px²). A single confidence threshold does not work optimally for all sizes.

**7. Background Imbalance**
Approximately 28% of training images contain no target annotations — pure background frames. This creates a classic foreground/background imbalance problem.

#### Augmentation Strategy

Each augmentation choice is directly justified by the dataset statistics above.

**Learning Rate & Optimization:**

| Parameter | Value | Purpose |
|---|---|---|
| `lr0` | 0.001 | Initial learning rate — step size for AdamW weight updates |
| `lrf` | 0.01 | Final LR multiplier; training ends at lr0 × lrf = 0.00001 |
| `momentum` | 0.937 | Uses 93.7% of previous gradient direction to smooth updates |
| `weight_decay` | 0.0005 | AdamW weight penalty preventing overfitting |
| `warmup_epochs` | 3 | LR ramps from near 0 to lr0 over first 3 epochs |

**Data Augmentation:**

| Augmentation | Value | Rationale |
|---|---|---|
| `mosaic` | 1.0 | 4-image stitching simulates crowded scenes — justified by 53 objects/image average |
| `mixup` | 0.15 | 15% transparent image blending balances car/human exposure |
| `hsv_h` | 0.015 | Random hue shift ±1.5% simulates lighting types |
| `hsv_s` | 0.7 | Random saturation ±70% simulates faded/vibrant conditions |
| `hsv_v` | 0.4 | Random brightness ±40% simulates shadows and glare |
| `fliplr` | 0.5 | 50% horizontal flip — drone imagery has no preferred orientation |
| `scale` | 0.5 | Random zoom ±50% — justified by 24,000× area range across altitudes |
| `translate` | 0.1 | 10% positional shift for border object learning |
| `degrees` | 0.0 | No rotation — aerial objects maintain consistent orientation |

**Prediction Thresholds:**

| Parameter | Value | Purpose |
|---|---|---|
| `conf` | 0.25 | Minimum 25% confidence to report a detection |
| `iou` | 0.45 | NMS threshold — overlapping boxes >45% overlap are suppressed |
| `max_det` | 300 | Maximum detections per image — justified by ~53 objects/image average |

#### Preprocessing Decisions Summary

| Observation from data | Decision | Parameter |
|---|---|---|
| Human median area = 304 px² | High resolution input | `imgsz=1280` |
| 7 irrelevant classes in dataset | Filter to targets only | `classes=[0,1,3]` |
| ~53 objects per image average | Dense scene augmentation | `mosaic=1.0` |
| Area range 3 → 72,670 px² | Scale variation training | `scale=0.5` |
| Up to 300+ objects per image | High detection limit | `max_det=300` |
| Cars 144k vs humans 106k | Class balance augmentation | `mixup=0.15` |

---

### Task 02 – Model Training

**Script:** `src/train.py`

Fine-tunes **YOLOv8n** (nano variant) using the Ultralytics framework on the VisDrone dataset.

#### Model Choice

| Property | Value |
|---|---|
| Model | YOLOv8n |
| Parameters | 3,012,798 |
| GFLOPs | 8.2 |
| Layers | 130 |
| Pretrained on | COCO (80 classes) |
| Fine-tuned on | VisDrone (classes 0, 1, 3) |

YOLOv8n was chosen for its speed/accuracy tradeoff on small aerial objects. At 71 FPS on a T4 GPU with imgsz=1280 it achieves strong mAP while remaining suitable for real-time deployment. The pretrained COCO weights provide a strong initialization — the model already knows what humans and cars look like, it just needs to learn to recognize them from above at tiny scale.

#### Training Configuration

| Parameter | Value |
|---|---|
| Base model | `yolov8n.pt` |
| Epochs | 50 |
| Image size | 1280 × 1280 |
| Batch size | 4 |
| Device | Tesla T4 GPU |
| Optimizer | AdamW |
| Learning rate | 0.001 → 0.00001 |
| Warmup epochs | 3 |
| Confidence threshold | 0.25 |
| IoU threshold | 0.45 |
| Max detections | 300 |
| Save period | Every 5 epochs |

#### Training Approach

The training proceeds in 5 stages each epoch:

```
For each batch of 4 images:
    ↓
Load 4 images → apply Mosaic (stitch 4 random images into one)
    ↓
15% chance: apply MixUp (blend with another image)
    ↓
Apply remaining augmentations (flip, scale, color, translate)
    ↓
Forward pass through 130 layers (all 4 images simultaneously)
    ↓
Loss calculated:
    box_loss  → how far off are the predicted box positions?
    cls_loss  → how often is the wrong class predicted?
    dfl_loss  → how accurately are the box edges placed?
    ↓
Backpropagation → AdamW updates 3M weights
    ↓
Next batch...
    ↓
After all 1,618 batches (one epoch complete):
Validate on 548 val images → compute mAP
    ↓
If best mAP so far → save as best.pt
```

**Learning rate schedule:**
```
Epochs 1–3   (warmup) : LR gradually increases  0 → 0.001
Epochs 3–50  (decay)  : LR gradually decreases  0.001 → 0.00001
```

The warmup prevents unstable large updates at the start. For lower training regime (50 epochs), a cosine curve won't have enough time to breathe. Lnear decay is much safer here. 

**Checkpoint saving:**

```python
save_period = 5   # save every 5 epochs
```

Checkpoints are saved every 5 epochs so training can resume from a recent epoch if interrupted — critical for long Kaggle sessions.

**Final weights:**
```
runs/train/visdrone_yolov8n/weights/
    ├── best.pt    ← highest validation mAP during training
    └── last.pt    ← final epoch weights
```

**Resuming interrupted training:**

```python
from ultralytics import YOLO
model   = YOLO("runs/train/visdrone_yolov8n/weights/last.pt")
results = model.train(resume=True, device="0")
```

#### Sample Predictions

Annotated prediction images are saved to `outputs/predictions/`. Each image shows bounding boxes with class labels and confidence scores, along with a semi-transparent count panel in the top-left corner displaying total humans and cars detected.

---

### Task 03 – Detection & Human Counting

**Script:** `src/detect.py`

Runs YOLOv8 inference on a single image, directory of images, or video file.

For each image/frame the pipeline:
1. Runs YOLOv8 inference (conf ≥ 0.25, IoU ≤ 0.45, classes [0, 1, 3])
2. Draws colour-coded bounding boxes — green for human (classes 0 and 1), blue for car (class 3) — with confidence labels
3. Counts human and car detections
4. Overlays the counts on a semi-transparent panel in the top-left corner
5. Saves the annotated result to `outputs/predictions/`

**Counting logic:**

```python
HUMAN_CLASSES = {0, 1}   # pedestrian + people → both counted as human
CAR_CLASSES   = {3}       # car

def count_objects(results) -> dict:
    counts = {"human": 0, "car": 0}
    for cls_id in results.boxes.cls.cpu().numpy().astype(int):
        if cls_id in HUMAN_CLASSES:
            counts["human"] += 1
        elif cls_id in CAR_CLASSES:
            counts["car"]   += 1
    return counts
```

**Sample detection results (8 validation images):**

| Image | Humans | Cars | FPS |
|---|---|---|---|
| 0000026_03000 | 0 | 10 | 1.1 |
| 0000289_03201 | 11 | 22 | 42.8 |
| 0000289_03601 | 22 | 16 | 49.7 |
| 0000291_01601 | 21 | 17 | 48.5 |
| 0000301_01001 | 7 | 24 | 49.7 |
| 0000312_00001 | 27 | 51 | 49.1 |
| 0000312_02201 | 26 | 34 | 49.6 |
| 0000359_03529 | 4 | 4 | 49.8 |
| **Total** | **118** | **178** | **42.5 avg** |

**Usage:**

```bash
python src/detect.py --weights runs/train/visdrone_yolov8n/weights/best.pt \
                     --source path/to/images/ \
                     --conf 0.25 --iou 0.45 \
                     --save_dir outputs/predictions
```

---



### Task 05 – Evaluation & Visualization

**Scripts:** `src/evaluate.py`, `src/visualize.py`

#### Quantitative Evaluation

Evaluated on the 548-image VisDrone validation set using Ultralytics `.val()`.

**Per-class results (target classes only):**

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| pedestrian | 520 | 8,844 | 0.719 | 0.484 | 0.439 | 0.224 |
| people | 482 | 5,125 | 0.694 | 0.391 | 0.333 | 0.141 |
| car | 515 | 14,064 | 0.836 | 0.801 | 0.772 | 0.563 |

**Overall metrics:**

| Metric | Value | Notes |
|---|---|---|
| mAP@0.5 (all 10 classes) | 0.154 | Dragged down by 7 untrained classes scoring 0.0 |
| mAP@0.5 (trained classes only) | 0.515 | True performance on pedestrian + people + car |
| Precision | 0.749 | |
| Recall | 0.558 | |
| Mean FPS | 71.1 | Well above 30 FPS real-time threshold |
| Median FPS | 75.7 | |
| Inference latency | 14.3 ms ± 2.0 ms | Per image on T4 GPU |

> **Note on overall mAP:** The 0.154 overall figure is averaged across all 10 VisDrone classes. The 7 classes not trained on (bicycle, van, truck, tricycle, awning-tricycle, bus, motor) all score 0.0 and heavily drag the average down. The per-class numbers for pedestrian (0.439), people (0.333), and car (0.772) represent the model's true performance.

#### Metric Explanations

- **mAP@0.5** — mean Average Precision at 50% IoU. A detection is correct if it overlaps the ground truth box by at least 50%. Higher is better.
- **mAP@0.5:0.95** — stricter COCO-style metric averaging across IoU thresholds 50%→95% in 5% steps.
- **Precision** — of all detections made, what fraction were actually correct? High precision = few false positives.
- **Recall** — of all real objects in the images, what fraction did the model find? High recall = few missed detections.
- **FPS** — 71.1 frames per second is well above the 30 FPS minimum needed for real-time drone video feeds.
- **Latency** — 14.3 ms per image means each frame is processed in under 15 milliseconds.

#### Visualizations Generated

| File | Description |
|---|---|
| `outputs/visualizations/class_distribution.png` | Annotation counts per target class |
| `outputs/visualizations/bbox_size_dist.png` | Bounding box area histograms |
| `outputs/visualizations/sample_annotations.png` | Ground truth boxes on 4 training images |
| `outputs/visualizations/training_curves.png` | Box loss, cls loss, DFL loss, and mAP per epoch |
| `outputs/visualizations/metrics_card.png` | Horizontal bar chart summarising all key metrics |
| `outputs/visualizations/prediction_grid.png` | 3×3 grid of annotated validation predictions |
| `outputs/visualizations/count_summary.png` | Per-image human and car counts across the sample set |
| `outputs/predictions/` | Individual annotated images with detection overlays |

---

## Results

**Training summary:**

| Setting | Value |
|---|---|
| Model | YOLOv8n |
| Epochs | 50 |
| Image size | 1280 |
| Batch size | 4 |
| Augmentation | mosaic=1.0, mixup=0.15, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, scale=0.5 |

**Evaluation results:**

| Metric | Value |
|---|---|
| Pedestrian mAP@0.5 | 0.439 |
| People mAP@0.5 | 0.333 |
| Car mAP@0.5 | 0.772 |
| Precision | 0.749 |
| Recall | 0.558 |
| Mean FPS | 71.1 |
| Inference latency | 14.3 ms |

Sample prediction outputs, training curves, and evaluation charts are saved in `outputs/visualizations/`.

---

## Strengths & Limitations

### Strengths

- **High-resolution inference** — imgsz=1280 preserves small object detail that is lost at the standard 640px resolution, critical for aerial datasets where humans are often under 20px tall
- **Strong car detection** — car mAP@0.5 of 0.772 is excellent; cars have consistent shapes and are larger than humans from aerial view
- **Real-time capable** — 71.1 FPS with 14.3 ms latency is well above the threshold for live drone feed processing
- **Mosaic + MixUp augmentation** — significantly improves robustness to varying crowd density and altitude-induced scale changes
- **DeepSORT tracking** — provides persistent object IDs without retraining; Re-ID embeddings prevent ID switches during occlusions
- **Modular architecture** — each task is an independent, importable Python module; any component can be used standalone
- **Checkpoint resilience** — save_period=5 means at most 5 epochs of training are lost on any session interruption

### Limitations

- **YOLOv8n is the smallest YOLO variant** — larger variants (s/m/l) would improve mAP significantly at the cost of FPS
- **Very small humans** — objects under 10 pixels remain difficult to detect reliably regardless of model size
- **Frame-level counting** — detect.py counts objects visible per frame, not unique objects across time; tracking is required for scene-level unique counts
- **Dense crowd occlusions** — at extreme crowd densities (100+ people per frame) overlapping proposals increase missed detections even after NMS tuning
- **GPU required for real-time** — CPU inference drops to ~1–2 FPS, making real-time deployment impractical without a GPU
- **Fixed confidence threshold** — a single conf=0.25 threshold may not work optimally across all drone altitudes

### Challenges Faced

- **Small object visibility** — VisDrone's median human bbox of ~304 px² is below the effective detection threshold at 640px, making imgsz=1280 mandatory but memory-intensive
- **Multi-GPU DDP crash** — distributed training across two T4 GPUs crashed at epoch 19 due to a PyTorch DDP incompatibility in Kaggle's environment; resolved by switching to single-GPU training
- **Session disconnections** — Kaggle session resets required explicit checkpoint saving every 5 epochs and immediate weight backup after training to prevent data loss
- **Class filtering** — including all 10 VisDrone classes caused the model to waste capacity on irrelevant classes; filtering to `classes=[0,1,3]` focused training on the target objects

---

## Demo Video

https://drive.google.com/drive/folders/1HGIn9ID62G4WV83xA0ebpPXeseOxB8j9


The demo covers:
- Dataset exploration and sample annotations
- Training run summary and loss/mAP curves
- Live inference with bounding boxes and human/car count overlay
- Tracking implementation walkthrough (`src/track.py`) — DeepSORT pipeline explanation and code review
