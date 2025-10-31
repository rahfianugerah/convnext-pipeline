#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive EDA for image datasets.

Works with:
1) Unsplit datasets:   root/<class>/*.jpg
2) Split datasets:     root/{train,val,validation,test}/<class>/*.jpg

Outputs to <output_dir>:
  - summary.txt
  - per_class_stats.json
  - corrupt.csv (if any)
  - files.csv (all scanned items with attributes)
  - figures/*.png  (all visualizations)

Usage (Python):
    from image_eda_report import generate_image_eda_report
    generate_image_eda_report(r"data\\cls", r"artifacts\\eda_cls", split_mode="auto", separate_split_reports=False)

Usage (CLI):
    python image_eda_report.py --dataset "data/cls" --out "artifacts/eda_cls"
    # If dataset already split and you want a report per split:
    python image_eda_report.py --dataset "data/cls" --out "artifacts/eda_cls" --split separate
"""

from __future__ import annotations
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Utility helpers
# -----------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}

def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def _safe_open(p: Path) -> Optional[Image.Image]:
    try:
        with Image.open(p) as im:
            im.load()
            return im.copy()
    except Exception:
        return None

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_fig(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def _colorfulness_fast(im: Image.Image) -> float:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

def _brightness(im: Image.Image) -> float:
    gs = ImageOps.grayscale(im)
    return float(np.asarray(gs, dtype=np.float32).mean())

def _aspect(w: int, h: int) -> float:
    return float(w) / float(h) if h else np.nan

def _detect_splits(root: Path) -> Dict[Path, str]:
    """Map file -> split if under known split directory names."""
    mapping = {}
    for sp_dir in root.iterdir():
        if not sp_dir.is_dir():
            continue
        name = sp_dir.name.lower()
        if name in SPLIT_NAMES:
            normalized = "train" if name == "train" else ("test" if name == "test" else "val")
            for p in sp_dir.rglob("*"):
                if _is_img(p):
                    mapping[p.resolve()] = normalized
    return mapping

# -----------------------
# Core scanning
# -----------------------
@dataclass
class ScanRow:
    path: str
    rel_path: str
    split: str
    label: str
    width: int
    height: int
    aspect: float
    file_size: int
    brightness: float
    colorfulness: float
    mode: str

def _scan_dataset(root: Path, recursive: bool = True) -> Tuple[pd.DataFrame, List[Path]]:
    split_map = _detect_splits(root)
    files: List[Path] = []
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if _is_img(p):
            files.append(p)

    rows: List[ScanRow] = []
    corrupt: List[Path] = []

    for p in files:
        im = _safe_open(p)
        if im is None:
            corrupt.append(p)
            continue

        w, h = im.size
        mode = im.mode
        sz = p.stat().st_size
        sp = split_map.get(p.resolve(), "")
        # determine class label
        # For split layout, label is the immediate parent under the split dir
        label = "__unlabeled__"
        if sp:
            # Expect .../<split>/<class>/<file>
            # If directly under split (no class dir) -> unlabeled
            parent = p.parent
            if parent.name.lower() not in SPLIT_NAMES:
                label = parent.name
        else:
            label = p.parent.name if p.parent != root else "__unlabeled__"

        rows.append(
            ScanRow(
                path=str(p.resolve()),
                rel_path=str(p.relative_to(root)),
                split=sp,
                label=label,
                width=w,
                height=h,
                aspect=_aspect(w, h),
                file_size=sz,
                brightness=_brightness(im),
                colorfulness=_colorfulness_fast(im),
                mode=mode,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df, corrupt

# -----------------------
# Visualizations
# -----------------------
def _bar_class_counts(df: pd.DataFrame, out: Path, title="Number of Images per Class"):
    counts = df["label"].value_counts().sort_values(ascending=False)
    fig = plt.figure(figsize=(max(6, 0.35 * len(counts) + 4), 4))
    idx = np.arange(len(counts))
    plt.bar(idx, counts.values)
    plt.xticks(idx, counts.index, rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    _save_fig(fig, out)

def _hist(values, bins, title, xlabel, out: Path):
    fig = plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    _save_fig(fig, out)

def _scatter(x, y, title, xlabel, ylabel, out: Path):
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _save_fig(fig, out)

def _grid_samples(df: pd.DataFrame, out: Path, n=30, cols=6, title="Random Sample of Images"):
    import random
    paths = df["path"].tolist()
    if not paths:
        return
    random.shuffle(paths)
    paths = paths[:n]
    rows = math.ceil(len(paths) / cols)
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
    fig.suptitle(title)
    for i, p in enumerate(paths, 1):
        im = _safe_open(Path(p))
        if im is None:
            continue
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(Path(p).parent.name, fontsize=8)
        ax.axis("off")
    _save_fig(fig, out)

def _per_class_grids(df: pd.DataFrame, out: Path, per_class=6, cols=6):
    from itertools import islice
    labels = df["label"].unique().tolist()
    if not labels:
        return
    samples: List[Path] = []
    for lb in labels[:8]:  # cap to keep grid reasonable
        sub = df[df["label"] == lb]
        take = min(per_class, len(sub))
        samples.extend(sub.sample(take, random_state=42)["path"].tolist())
    if not samples:
        return
    rows = math.ceil(len(samples) / cols)
    fig = plt.figure(figsize=(cols * 2.0, rows * 2.0))
    fig.suptitle("Per-class Examples")
    for i, p in enumerate(samples, 1):
        im = _safe_open(Path(p))
        if im is None:
            continue
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(Path(p).parent.name, fontsize=8)
        ax.axis("off")
    _save_fig(fig, out)

def _color_mode_bar(df: pd.DataFrame, out: Path):
    counts = df["mode"].value_counts()
    fig = plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(counts)), counts.values)
    plt.xticks(np.arange(len(counts)), counts.index)
    plt.title("Image Color Modes")
    plt.xlabel("PIL mode")
    plt.ylabel("Count")
    _save_fig(fig, out)

def _per_class_avg_rgb_hist(df: pd.DataFrame, out_dir: Path, bins=256):
    """For each class, compute mean RGB histogram and save plot."""
    classes = sorted([c for c in df["label"].unique().tolist() if c != "__unlabeled__"])
    for cls in classes:
        sub = df[df["label"] == cls]
        if sub.empty:
            continue
        # accumulate histograms
        hist_r = np.zeros(bins, dtype=np.float64)
        hist_g = np.zeros(bins, dtype=np.float64)
        hist_b = np.zeros(bins, dtype=np.float64)
        n = 0
        for p in sub["path"].tolist():
            im = _safe_open(Path(p))
            if im is None:
                continue
            arr = np.asarray(im.convert("RGB"))
            hr, _ = np.histogram(arr[..., 0], bins=bins, range=(0, 255))
            hg, _ = np.histogram(arr[..., 1], bins=bins, range=(0, 255))
            hb, _ = np.histogram(arr[..., 2], bins=bins, range=(0, 255))
            hist_r += hr
            hist_g += hg
            hist_b += hb
            n += 1
        if n == 0:
            continue
        hist_r /= n
        hist_g /= n
        hist_b /= n
        x = np.arange(bins)
        fig = plt.figure(figsize=(7, 4))
        plt.plot(x, hist_r, label="R")
        plt.plot(x, hist_g, label="G")
        plt.plot(x, hist_b, label="B")
        plt.title(f"Average RGB Histogram – {cls}")
        plt.xlabel("Intensity (0–255)")
        plt.ylabel("Average count")
        plt.legend()
        _save_fig(fig, out_dir / f"rgb_hist_{cls}.png")

def _per_class_brightness_hist(df: pd.DataFrame, out_dir: Path, bins=40):
    classes = sorted([c for c in df["label"].unique().tolist()])
    for cls in classes:
        sub = df[df["label"] == cls]
        if sub.empty:
            continue
        fig = plt.figure(figsize=(6, 4))
        plt.hist(sub["brightness"].values, bins=bins)
        plt.title(f"Brightness Distribution – {cls}")
        plt.xlabel("Average grayscale value (0–255)")
        plt.ylabel("Count")
        _save_fig(fig, out_dir / f"brightness_{cls}.png")

# -----------------------
# Public API
# -----------------------
def generate_image_eda_report(
    dataset_path: str,
    output_dir: str,
    *,
    split_mode: str = "auto",               # "auto" | "merged" | "separate"
    bins: int = 256,
    max_random_samples: int = 30
) -> None:
    """
    Perform EDA on an image dataset and save plots + summary.

    Parameters
    ----------
    dataset_path : str
        Root folder of the dataset (string is normalized to Path safely).
    output_dir : str
        Destination directory for outputs (created if missing).
    split_mode : str
        - "auto": detect splits; if present, merge to a single report.
        - "merged": ignore split boundaries, single report over all files.
        - "separate": if splits exist, produce one sub-report per split
          under output_dir/{train,val,test}. If no splits, falls back to one report.
    bins : int
        Number of bins for color histograms.
    max_random_samples : int
        How many random images to show on the samples grid.
    """
    root = Path(dataset_path)
    out_root = Path(output_dir)
    _ensure_dir(out_root)

    # Scan once for global metadata
    df_all, corrupt = _scan_dataset(root)
    if df_all.empty:
        (out_root / "summary.txt").write_text(f"No images found under: {root}\n", encoding="utf-8")
        return

    if corrupt:
        pd.DataFrame({"path": [str(p) for p in corrupt]}).to_csv(out_root / "corrupt.csv", index=False)
    df_all.to_csv(out_root / "files.csv", index=False)

    has_splits = df_all["split"].ne("").any()
    do_separate = (split_mode == "separate") and has_splits

    splits_to_run = ["ALL"]
    if do_separate:
        splits_to_run = sorted(df_all["split"].replace({"": "unspecified"}).unique().tolist())

    for split_name in splits_to_run:
        if split_name == "ALL":
            df = df_all.copy()
            sub_out = out_root
        else:
            df = df_all[df_all["split"].replace({"": "unspecified"}) == split_name].copy()
            sub_out = out_root / split_name
            _ensure_dir(sub_out)

        figs = sub_out / "figures"
        _ensure_dir(figs)

        # --- BASIC STATS ---
        n_total = len(df)
        classes = sorted([c for c in df["label"].unique().tolist()])
        n_classes = len(classes)
        class_counts = df["label"].value_counts().sort_values(ascending=False)

        # --- SAVE KEY FIGURES ---
        _bar_class_counts(df, figs / "class_counts.png")
        _hist(df["width"].values, 40, "Image Widths", "Width (px)", figs / "hist_width.png")
        _hist(df["height"].values, 40, "Image Heights", "Height (px)", figs / "hist_height.png")
        _hist(df["aspect"].values[~np.isnan(df["aspect"].values)], 40, "Aspect Ratios (W/H)", "Aspect ratio", figs / "hist_aspect.png")
        _hist((df["file_size"].values / 1024.0), 40, "File Sizes", "KB", figs / "hist_filesize.png")
        _hist(df["brightness"].values, 40, "Brightness (Average grayscale)", "0–255", figs / "hist_brightness.png")
        _scatter(df["width"].values, df["height"].values, "Width vs Height", "Width", "Height", figs / "scatter_wh.png")
        _color_mode_bar(df, figs / "color_modes.png")
        _grid_samples(df, figs / "random_samples.png", n=max_random_samples)
        _per_class_grids(df, figs / "per_class_grid.png", per_class=6, cols=6)
        _per_class_avg_rgb_hist(df, figs, bins=bins)
        _per_class_brightness_hist(df, figs, bins=40)

        # --- PER-CLASS STATS JSON ---
        per_class_stats = {}
        for cls, sub in df.groupby("label"):
            per_class_stats[cls] = {
                "count": int(len(sub)),
                "width_mean": float(sub["width"].mean()),
                "height_mean": float(sub["height"].mean()),
                "aspect_mean": float(sub["aspect"].mean()),
                "filesize_kb_mean": float((sub["file_size"]/1024.0).mean()),
                "brightness_mean": float(sub["brightness"].mean()),
                "colorfulness_mean": float(sub["colorfulness"].mean()),
                "modes": sub["mode"].value_counts().to_dict()
            }
        (sub_out / "per_class_stats.json").write_text(json.dumps(per_class_stats, indent=2), encoding="utf-8")

        # --- SUMMARY TXT ---
        summary = []
        hdr = f"Dataset path: {root}"
        if split_name != "ALL":
            hdr += f"  [split: {split_name}]"
        summary.append(hdr)
        summary.append(f"Total images: {n_total}")
        summary.append(f"Number of classes: {n_classes}\n")
        summary.append("Class distribution:")
        for k, v in class_counts.items():
            summary.append(f"  {k}: {v}")
        summary.append("")
        summary.append("Image dimension stats (pixels):")
        summary.append(str(df[["width", "height"]].describe()))
        summary.append("\nAspect ratio stats:")
        summary.append(str(df["aspect"].describe()))
        summary.append("\nFile size (KB) stats:")
        summary.append(str((df["file_size"]/1024.0).describe()))
        summary.append("\nBrightness stats (0–255):")
        summary.append(str(df["brightness"].describe()))
        summary.append("\nColor modes:")
        summary.append(str(df["mode"].value_counts()))
        summary.append("\nFigures saved in: " + str(figs))
        (sub_out / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


# -----------------------
# CLI
# -----------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Image dataset EDA")
    ap.add_argument("--dataset", required=True, help="Path to dataset root (use raw string or forward slashes on Windows)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--split", choices=["auto", "merged", "separate"], default="auto",
                    help="auto=detect and merge; merged=ignore splits; separate=report per split")
    ap.add_argument("--bins", type=int, default=256, help="Bins for RGB histograms")
    ap.add_argument("--max_samples", type=int, default=30, help="Max random samples to show")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    generate_image_eda_report(args.dataset, args.out, split_mode=args.split, bins=args.bins, max_random_samples=args.max_samples)
    print(f"[DONE] EDA Done - See: {Path(args.out).resolve()}")