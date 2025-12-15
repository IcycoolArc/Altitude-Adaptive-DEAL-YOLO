# Altitude-Adaptive Augmentation for DEAL-YOLO on WAID

This repository contains the code used for our ECE 4990 final project:

**Altitude-Adaptive Augmentation for DEAL–YOLO in UAV Wildlife Detection**

We implement a pseudo-altitude–driven augmentation framework (AAA) on top of the DEAL–YOLO detector and evaluate it on the WAID wildlife aerial imagery dataset. Our method improves detection performance on mid- and high-altitude views without modifying the underlying detector architecture.

---

## 1. Project Overview

### Motivation

UAV-based wildlife monitoring experiences strong altitude-dependent image quality changes:

- At low altitude, animals appear large and clearly detailed.
- At mid altitude, blur, atmospheric distortion, and compression artifacts appear.
- At high altitude, animals occupy only a few pixels and become extremely difficult to detect.

Standard YOLO augmentation is altitude-agnostic and therefore fails to model these real-world degradations, leading to reduced recall on high-altitude views.

### Main Idea: Altitude-Adaptive Augmentation (AAA)

The WAID dataset does not provide flight altitude metadata. To address this, we infer a pseudo-altitude label for each image by computing the mean normalized bounding-box area.

- Large bounding boxes → low altitude  
- Medium bounding boxes → mid altitude  
- Small bounding boxes → high altitude  

AAA then applies three corresponding augmentation regimes:

- Low altitude: mild geometric/photometric changes  
- Mid altitude: blur, noise, moderate downscaling  
- High altitude: strong downscaling and degradation  

Training proceeds in a curriculum-like sequence from low → mid → high altitude augmentations.

---

## 2. Repository Structure

```
Altitude-Adaptive-DEAL-YOLO/
│
├── README.md
├── .gitignore
│
├── configs/
│   └── data.yaml                     # WAID dataset definition (Roboflow export)
│
├── src/
│   ├── altitude_aug.py               # AAA implementation
│   ├── resume.py                     # Resume training script
│   └── train_aaa.py                  # Optional wrapper for AAA training
│
├── figures/
│   ├── tensorboard_losses.png        # Training and validation loss curves
│   └── tensorboard_lr_metrics.png    # mAP, precision, recall curves
│
├── paper/
│   └── Altitude_Adaptive_DEAL_YOLO.pdf
│
└── (local only – ignored by Git)
    └── runs/                         # YOLO auto-generated training logs and weights
```

---

## 3. Installation

Install dependencies:

```
pip install ultralytics opencv-python numpy albumentations
```

Clone the repository:

```
git clone https://github.com/<your-username>/Altitude-Adaptive-DEAL-YOLO.git
cd Altitude-Adaptive-DEAL-YOLO
```

---

## 4. Dataset Setup (WAID)

The WAID dataset is not provided in this repository.  
Download from Roboflow:

https://universe.roboflow.com/waid/waid-project-tciau/dataset/1

Expected folder structure:

```
datasets/
  waid/
    train/
      images/
      labels/
    valid/
      images/
      labels/
    test/
      images/
      labels/
    data.yaml
```

Copy WAID's `data.yaml` into:

```
configs/data.yaml
```

---

## 5. Running Altitude-Adaptive Augmentation (AAA)

To apply AAA and generate augmented images:

```
python src/altitude_aug.py
```

This script computes pseudo-altitude, assigns augmentation regimes, and writes augmented samples back into the dataset.

---

## 6. Training

### Baseline Training (no AAA)

```
yolo train model=yolov8m-ASF-P2.pt data=configs/data.yaml imgsz=640 epochs=300
```

### AAA-Enhanced Training

```
python src/train_aaa.py
```

Or manually:

```python
from ultralytics import YOLO
model = YOLO("yolov8m-ASF-P2.pt")
model.train(data="configs/data.yaml", epochs=300)
```

Note: Model weights are not in this repository and will be auto-downloaded by Ultralytics.

---

## 7. Resuming Training

To resume from the last saved checkpoint:

```
python src/resume.py
```

This script loads:

```
runs/train/<run-name>/weights/last.pt
```

---

## 8. Viewing Metrics

Use TensorBoard:

```
tensorboard --logdir runs/train
```

Representative images are located in:

```
figures/
```

---

## 9. Model Notes

This project uses the YOLOv8m-ASF-P2 model definition, consistent with the scaling described in the DEAL-YOLO paper.  
The Ultralytics engine does not implement DEAL-YOLO’s custom modules (LDConv, SSFF, DFA loss).  
AAA provides a data-centric enhancement compatible with any YOLO-based detector.

---

## 10. Paper

The full research paper is located in:

```
paper/Altitude_Adaptive_DEAL_YOLO.pdf
```

---

## 11. Contributors (ECE 4990)

Andrew Ravadan Castillo
Dia Agrawal
Izel Trejo

ECE 4990 – Department of Electrical and Computer Engineering  
Fall 2025
