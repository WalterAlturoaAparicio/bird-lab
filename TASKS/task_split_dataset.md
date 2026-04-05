# TASK.md

## Current Task

Implement dataset splitting for the bird classification pipeline.

---

## Context

This project processes bird images through:

1. Ingestion (zip → raw dataset)
2. YOLO-based cropping
3. Quality filtering:

   * accepted
   * needs_review
   * rejected

Only `accepted/` images should be used for dataset splitting.

---

## Requirements

* Split dataset into:

  * train (75%)
  * validation (15%)
  * test (10%)

* Must be:

  * stratified by class
  * balanced as much as possible

* Input:

  * `data/processed/accepted/<class_name>/*.jpg`

* Output:

  ```
  data/split/
    ├── train/
    ├── val/
    └── test/
  ```

Each split must preserve class subfolders.

---

## Constraints

* Do NOT duplicate images across splits
* Do NOT modify original files (copy or symlink)
* Maintain reproducibility (use random seed)
* Handle edge cases:

  * classes with very few samples
  * rounding issues in splits

---

## Edge Cases

* If class has < 10 images:

  * ensure at least 1 image in train
  * distribute remaining carefully

* If split ratios produce decimals:

  * prioritize train > val > test

---

## Expected Behavior

* Each class appears in all splits (if possible)
* Total distribution approximates 75/15/10
* Output directory is clean and consistent

---

## Suggested Implementation

* `src/utils/split.py`
* function:

  ```
  split_dataset(input_dir, output_dir, ratios, seed)
  ```

---

## Success Criteria

* Dataset is properly split
* No data leakage between splits
* Class distribution is preserved
* Script can be re-run deterministically

---

## Instructions for AI

* Focus on clean, modular code
* Avoid overengineering
* Use standard Python libraries unless necessary
* Include basic logging for verification
* Ensure code is easy to integrate into pipeline

---
