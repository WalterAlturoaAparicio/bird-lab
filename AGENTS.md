# AGENTS.md

## Project Overview

This project builds a complete pipeline for bird image classification, focusing on dataset quality, reproducibility, and iterative improvement.

The system processes raw image datasets into clean, structured, and model-ready data using automated and human-in-the-loop techniques.

---

## Core Objectives

* Build a robust dataset pipeline for bird classification
* Ensure high-quality image preprocessing (cropping, filtering)
* Enable iterative dataset improvement through feedback loops
* Maintain reproducibility and scalability

---

## Pipeline Stages

### 1. Ingestion

* Input: `.zip` files with class-based folders
* Output: structured dataset in `data/raw/`
* Handles:

  * image extraction
  * format normalization (converted to `.jpg`)
  * hash-based renaming
  * duplicate prevention

---

### 2. Preprocessing (YOLO Cropping)

* Uses YOLO for bird detection
* Applies dynamic margin to bounding boxes
* Clips to image boundaries (no padding allowed)
* Converts images to consistent format and size

---

### 3. Quality Analysis & Classification

Each image is classified into:

* `accepted/` → ready for training
* `needs_review/` → requires manual validation
* `rejected/` → discarded automatically

Based on:

* confidence score
* bounding box size ratio
* edge proximity
* aspect ratio
* clipping
* final image size

---

### 4. Human-in-the-loop Review

* Manual correction of `needs_review` images
* Tools: CVAT or similar annotation tools
* Corrected images are moved to `accepted/`

---

### 5. Dataset Split (Upcoming)

* Train / Validation / Test split (75/15/10)
* Only uses `accepted/` images
* Must maintain class balance

---

### 6. Training & Evaluation (Planned)

* Model selection (e.g., MobileNet, EfficientNet)
* Training with augmentation and checkpoints
* Metrics:

  * F1 score
  * precision / recall
  * confusion matrix

---

## Design Principles

* Prefer data quality over model complexity
* Avoid introducing artificial artifacts (e.g., black padding)
* Use reproducible pipelines instead of manual steps
* Keep logic modular (`src/`) and orchestration in notebooks
* Automate everything that can be automated
* Use manual review only where necessary

---

## Constraints & Rules

* Do NOT introduce black padding in images
* Always preserve aspect ratio when resizing
* Never trust YOLO output blindly — always validate
* Do NOT mix dataset stages (raw, processed, split)
* Avoid manual file manipulation outside the pipeline

---

## Metadata

Each processed image should store:

* original filename
* new filename (hash-based)
* class label
* confidence score
* bbox area ratio
* processing status (accepted / needs_review / rejected)

---

## Current Status

* ✅ Ingestion pipeline completed
* ✅ YOLO-based cropping implemented
* ✅ Quality filtering and classification system implemented
* ✅ Dataset split pending
* ⏳ Training pipeline pending

---

## Next Steps

1. Train baseline model
2. Identify weak classes via metrics
3. Review `needs_review` selectively
4. Iterate dataset improvements

---

## Guidelines for AI Assistants

When working on this project:

* Prioritize pipeline consistency over quick fixes
* Do not suggest manual solutions if automation is possible
* Ensure all steps are reproducible
* Always consider dataset quality impact before model changes
* Suggest improvements incrementally, not full rewrites

---
