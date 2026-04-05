"""Utilities for deterministic dataset splitting."""

from __future__ import annotations

import logging
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union


LOGGER = logging.getLogger(__name__)

# Emit a NullHandler so callers that don't configure logging don't see
# "No handlers could be found" warnings (logging best practice).
LOGGER.addHandler(logging.NullHandler())

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.75, 0.15, 0.10)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def split_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
    use_symlinks: bool = False,
) -> dict:
    """
    Split accepted images into train/val/test while preserving class folders.

    Args:
        input_dir:    Directory with class subfolders and image files.
        output_dir:   Target root where train/val/test folders will be created.
        ratios:       Split ratios ordered as (train, val, test).
        seed:         Random seed for deterministic shuffling.
        use_symlinks: If True, create symlinks instead of copying files.
                      Falls back to copy if the OS does not support symlinks.

    Returns:
        Summary dictionary with totals and per-class distribution.

    Raises:
        FileNotFoundError: If input_dir does not exist.
        ValueError:        If ratios are invalid or no images are found.
        RuntimeError:      If a duplicate target path is detected (internal guard).
    """
    in_path  = Path(input_dir)
    out_path = Path(output_dir)

    _validate_inputs(in_path, out_path, ratios)
    ratio_tuple = _normalize_ratios(ratios)

    class_files = _collect_class_images(in_path)
    if not class_files:
        raise ValueError(f"No class images found in: {in_path}")

    _prepare_output_dir(out_path)
    _create_split_folders(out_path)

    rng = random.Random(seed)
    per_class_counts: Dict[str, Dict[str, int]] = {}
    total_counts: Dict[str, int] = {split: 0 for split in SPLIT_NAMES}
    all_written_targets: Set[Path] = set()

    for class_name in sorted(class_files):
        files = list(class_files[class_name])
        rng.shuffle(files)

        counts      = _allocate_counts(len(files), ratio_tuple)
        assignments = _assign_files(files, counts)

        for split_name, split_files in assignments.items():
            target_class_dir = out_path / split_name / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for src in split_files:
                dst = target_class_dir / src.name
                if dst in all_written_targets:
                    raise RuntimeError(
                        f"Duplicate target assignment detected: {dst}"
                    )
                _link_or_copy(src, dst, use_symlinks=use_symlinks)
                all_written_targets.add(dst)

            total_counts[split_name] += len(split_files)

        per_class_counts[class_name] = {
            name: len(assignments[name]) for name in SPLIT_NAMES
        }
        LOGGER.info(
            "Class '%s' (%d total): train=%d val=%d test=%d",
            class_name,
            len(files),
            per_class_counts[class_name]["train"],
            per_class_counts[class_name]["val"],
            per_class_counts[class_name]["test"],
        )

    summary = {
        "input_dir":   str(in_path),
        "output_dir":  str(out_path),
        "ratios":      {
            "train": ratio_tuple[0],
            "val":   ratio_tuple[1],
            "test":  ratio_tuple[2],
        },
        "seed":        seed,
        "use_symlinks": use_symlinks,
        "totals":      total_counts,
        "per_class":   per_class_counts,
        "num_classes": len(per_class_counts),
        "num_images":  sum(total_counts.values()),
    }
    LOGGER.info(
        "Split complete: classes=%d  total=%d  train=%d  val=%d  test=%d",
        summary["num_classes"],
        summary["num_images"],
        total_counts["train"],
        total_counts["val"],
        total_counts["test"],
    )
    return summary


# ── Validation ────────────────────────────────────────────────────

def _validate_inputs(
    input_dir: Path,
    output_dir: Path,
    ratios: Sequence[float],
) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    # Guard against accidentally pointing output to input
    if output_dir.exists() and output_dir.resolve() == input_dir.resolve():
        raise ValueError(
            "output_dir must be different from input_dir to avoid data loss."
        )
    if len(ratios) != 3:
        raise ValueError("Expected exactly 3 ratios: (train, val, test).")
    if any(r < 0 for r in ratios):
        raise ValueError("All ratios must be non-negative.")
    if sum(ratios) <= 0:
        raise ValueError("Ratios must sum to a value greater than 0.")


# ── Ratio helpers ─────────────────────────────────────────────────

def _normalize_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    total = float(sum(ratios))
    return (ratios[0] / total, ratios[1] / total, ratios[2] / total)


# ── File collection ───────────────────────────────────────────────

def _collect_class_images(input_dir: Path) -> Dict[str, List[Path]]:
    classes: Dict[str, List[Path]] = {}
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            classes[class_dir.name] = images
    return classes


# ── Directory management ──────────────────────────────────────────

def _prepare_output_dir(output_dir: Path) -> None:
    """Wipe and recreate the output directory.

    The caller must ensure output_dir != input_dir (enforced in _validate_inputs).
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _create_split_folders(output_dir: Path) -> None:
    for split_name in SPLIT_NAMES:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)


# ── Allocation ────────────────────────────────────────────────────

def _allocate_counts(
    n: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, int]:
    """
    Distribute *n* images across train/val/test.

    Guarantees:
    - sum(counts) == n  always.
    - counts["train"] >= 1  for every non-empty class.
    - counts["val"] >= 1 and counts["test"] >= 1  when n >= 3.
    """
    if n <= 0:
        return {"train": 0, "val": 0, "test": 0}

    raw    = {"train": n * ratios[0], "val": n * ratios[1], "test": n * ratios[2]}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}

    # Distribute floor remainder left-to-right (train > val > test).
    remainder = n - sum(counts.values())
    for name in ("train", "val", "test"):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1

    # Guarantee at least one training sample.
    if counts["train"] == 0:
        donor = _largest_bucket(counts, exclude="train")
        if donor and counts[donor] > 0:
            counts[donor] -= 1
            counts["train"] += 1

    # Guarantee val and test each have at least one sample when possible.
    if n >= 3:
        for split_name in ("val", "test"):
            if counts[split_name] == 0:
                donor = _largest_bucket(counts, exclude=split_name)
                if donor and counts[donor] > 1:
                    counts[donor] -= 1
                    counts[split_name] += 1

    if sum(counts.values()) != n:
        raise RuntimeError(
            f"Allocation mismatch for class size {n}: {counts}"
        )
    return counts


def _largest_bucket(
    counts: Dict[str, int],
    exclude: str,
) -> Optional[str]:
    candidates = [(k, v) for k, v in counts.items() if k != exclude]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


# ── File assignment ───────────────────────────────────────────────

def _assign_files(
    files: Sequence[Path],
    counts: Dict[str, int],
) -> Dict[str, List[Path]]:
    train_n = counts["train"]
    val_n   = counts["val"]

    if train_n + val_n + counts["test"] != len(files):
        raise RuntimeError("Assigned counts do not match number of files.")

    return {
        "train": list(files[:train_n]),
        "val":   list(files[train_n: train_n + val_n]),
        "test":  list(files[train_n + val_n:]),
    }


# ── I/O ───────────────────────────────────────────────────────────

def _link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    """Symlink *src* to *dst*, falling back to a plain copy on failure."""
    if use_symlinks:
        try:
            dst.symlink_to(src.resolve())
            return
        except (OSError, NotImplementedError):
            LOGGER.warning(
                "Symlink creation failed for %s -> %s; falling back to copy.",
                src, dst,
            )
    shutil.copy2(src, dst)
