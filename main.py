#!/usr/bin/env python3
"""
ConvNeXt + Optuna Multi-Task Pipeline (Train/Val/Test) with Standard Metrics

Dependencies +  Run Instructions:
  conda create -n convnext python=3.10 -y
  conda activate convnext
  conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
  conda install -y -c conda-forge optuna opencv
  pip install timm scikit-learn torchmetrics jiwer

Data layouts:
  classification:
    data_root/
      train/<class>/*.jpg
      val/<class>/*.jpg
      test/<class>/*.jpg

  object detection:
    data_root/
      images/*.jpg
      train.csv
      val.csv
      test.csv
    # CSV columns: filename,xmin,ymin,xmax,ymax,label

  ocr (word recognition):
    data_root/
      train/images/*.png
      train/labels.txt   # img001.png<TAB>HELLO
      val/images/*.png
      val/labels.txt
      test/images/*.png
      test/labels.txt
"""

# Manual ConvNeXt Variant Switch
# Choose one: "convnext_tiny", "convnext_small", "convnext_base" (the bigger the longer training time/memory)
MANUAL_CONVNEXT_VARIANT = "convnext_tiny"

# Importing libraries
import os
import cv2
import json
import time
import timm
import optuna
import argparse
import numpy as np

# Additional imports for dataset handling, Kaggle support, and plotting
import urllib.request
import zipfile
import tarfile
import shutil
import subprocess
import re
import glob
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torchvision
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from tqdm import tqdm

# New AMP API (PyTorch 2.5+)
from torch import amp
SCALER = amp.GradScaler('cuda', enabled=True)

# Metrics libs
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion
from jiwer import wer, cer

# Speed knobs (CUDA)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_image_cv2(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

# Transforms
class ClassificationTransform:
    def __init__(self, size=224, train=True):
        self.size = size
        self.train = train

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.train:
            h, w = img.shape[:2]
            scale = np.random.uniform(0.7, 1.0)
            nh, nw = int(h * scale), int(w * scale)
            y0 = np.random.randint(0, max(1, h - nh + 1))
            x0 = np.random.randint(0, max(1, w - nw + 1))
            img = img[y0:y0+nh, x0:x0+nw]
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.randint(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        t = TF.to_tensor(Image.fromarray(img))
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t

class ObjectDetectionTransform:
    def __init__(self, size=640, train=True):
        self.size = size
        self.train = train

    def __call__(self, img_bgr: np.ndarray, boxes: np.ndarray):
        h, w = img_bgr.shape[:2]
        if self.train and np.random.rand() < 0.5:
            img_bgr = img_bgr[:, ::-1, :]
            if boxes.size > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        scale = min(self.size / h, self.size / w)
        nh, nw = int(h * scale), int(w * scale)
        img_res = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        top = (self.size - nh) // 2
        left = (self.size - nw) // 2
        canvas[top:top+nh, left:left+nw] = img_res

        if boxes.size > 0:
            boxes = boxes.copy().astype(np.float32)
            boxes *= scale
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top

        t = TF.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t, boxes

class OCRTransform:
    def __init__(self, img_h=32, img_w=256, train=True):
        self.img_h = img_h
        self.img_w = img_w
        self.train = train

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=15, C=10
        )
        if self.train and np.random.rand() < 0.3:
            h, w = bin_img.shape
            pts1 = np.float32([[0,0],[w-1,0],[0,h-1]])
            dx = np.random.uniform(-0.02*w, 0.02*w)
            dy = np.random.uniform(-0.02*h, 0.02*h)
            pts2 = np.float32([[dx,0],[w-1+dx,dy],[0,h-1+dy]])
            M = cv2.getAffineTransform(pts1, pts2)
            bin_img = cv2.warpAffine(bin_img, M, (w, h), borderValue=255)

        h, w = bin_img.shape
        scale = min(self.img_h / h, self.img_w / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(bin_img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.img_h, self.img_w), 255, dtype=np.uint8)
        top = (self.img_h - nh) // 2
        left = (self.img_w - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        t = TF.to_tensor(Image.fromarray(canvas))
        t = TF.normalize(t, mean=[0.5], std=[0.5])
        return t

# Datasets
class ClassificationDataset(Dataset):
    """
    A generic image classification dataset that reads images from
    a directory structure of the form <root>/<split>/<class>/<image>.
    An optional class_to_idx mapping can be provided to ensure that
    all splits use the exact same class-index ordering. If not
    provided, it will infer the mapping from the subdirectories under
    the given split.
    """
    def __init__(self, root: str, split: str, transform: ClassificationTransform,
                 class_to_idx: Dict[str, int] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        # Determine class mapping
        if class_to_idx is None:
            # Infer class names from directory names under the split
            classes = sorted([p.name for p in (self.root / split).iterdir() if p.is_dir()])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            # Use provided mapping (copy to avoid side effects)
            self.class_to_idx = dict(class_to_idx)
            # Ensure that every class directory exists under this split, even if empty
            for c in self.class_to_idx.keys():
                (self.root / split / c).mkdir(parents=True, exist_ok=True)
        # Collect sample paths and labels
        for c, idx in self.class_to_idx.items():
            class_dir = self.root / split / c
            if not class_dir.exists():
                continue
            for imgp in class_dir.glob("*.*"):
                if imgp.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.samples.append((str(imgp), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = read_image_cv2(path)
        x = self.transform(img)
        return x, label

class ObjectDetectionDataset(Dataset):
    def __init__(self, root: str, split: str, transform: ObjectDetectionTransform):
        self.root = Path(root)
        self.transform = transform
        self.csv_path = self.root / f"{split}.csv"
        self.img_dir = self.root / "images"
        rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("filename"):
                    continue
                filename, xmin, ymin, xmax, ymax, label = line.split(",")
                rows.append((filename, float(xmin), float(ymin), float(xmax), float(ymax), label))
        self.index = {}
        labels_set = set()
        for r in rows:
            labels_set.add(r[5])
            self.index.setdefault(r[0], []).append(r[1:])
        self.classes = sorted(list(labels_set))
        self.class_to_idx = {c: i+1 for i, c in enumerate(self.classes)}
        self.files = sorted(list(self.index.keys()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = str(self.img_dir / fname)
        img = read_image_cv2(img_path)

        recs = self.index[fname]
        if len(recs) == 0:
            boxes = np.zeros((0,4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = []
            labels = []
            for xmin, ymin, xmax, ymax, lab in recs:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[lab])
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        x, boxes = self.transform(img, boxes)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([idx], dtype=torch.int64)}
        return x, target

def detection_collate(batch):
    imgs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    return torch.stack(imgs), tgts

class OCRLabelEncoder:
    def __init__(self, charset: str):
        self.charset = charset
        self.idx2ch = {i+1: ch for i, ch in enumerate(charset)}
        self.ch2idx = {ch: i+1 for i, ch in enumerate(charset)}

    @property
    def vocab_size(self):
        return len(self.charset) + 1

    def encode(self, text: str) -> List[int]:
        return [self.ch2idx[ch] for ch in text if ch in self.ch2idx]

    def decode_greedy(self, logits: torch.Tensor) -> List[str]:
        probs = logits.detach().cpu().softmax(-1)
        seq = probs.argmax(-1)  # T x B
        T, N = seq.shape
        results = []
        for i in range(N):
            prev = -1
            out = []
            for t in range(T):
                s = int(seq[t, i])
                if s != prev and s != 0:
                    out.append(self.idx2ch.get(s, ""))
                prev = s
            results.append("".join(out))
        return results

class OCRDataset(Dataset):
    def __init__(self, root: str, split: str, transform: OCRTransform, charset: str):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_dir = self.root / split / "images"
        label_file = self.root / split / "labels.txt"
        self.samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                fname, text = line.split("\t", 1)
                self.samples.append((str(self.img_dir / fname), text))
        self.encoder = OCRLabelEncoder(charset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = read_image_cv2(path)
        x = self.transform(img)
        y = torch.tensor(self.encoder.encode(text), dtype=torch.long)
        return x, y, len(y)

def ocr_collate(batch):
    imgs, ys, ylens = zip(*batch)
    x = torch.stack(imgs)
    target = torch.cat(ys, dim=0)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return x, target, target_lengths

###########################################################################
# Dataset preparation helpers (generic, classification, detection, OCR)
# and Kaggle dataset download support.
###########################################################################

# Regular expression to detect Kaggle dataset URLs
_KAGGLE_RE = re.compile(
    r"^https?://(?:www\.)?kaggle\.com/datasets/([^/]+)/([^/?#]+)(?:/.*)?$",
    re.IGNORECASE,
)

def is_kaggle_url(url: str) -> bool:
    """Return True if the URL corresponds to a Kaggle dataset page."""
    return bool(_KAGGLE_RE.match(url or ""))

def kaggle_slug(url: str) -> str:
    """Extract owner/slug from a Kaggle dataset URL."""
    m = _KAGGLE_RE.match(url or "")
    if not m:
        return ""
    owner = m.group(1).strip()
    slug = m.group(2).strip()
    return f"{owner}/{slug}"

def ensure_kaggle_cli_available():
    """Verify that the Kaggle CLI is installed and accessible."""
    from shutil import which
    if which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install with 'pip install kaggle' and place "
            "your API token at ~/.kaggle/kaggle.json or %USERPROFILE%\\.kaggle\\kaggle.json."
        )

def kaggle_download_dataset(slug: str, download_dir: Path):
    """
    Use the Kaggle CLI to download a dataset given its slug (owner/slug). The
    downloaded zip(s) are placed in download_dir. Returns the list of downloaded
    archive paths.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(download_dir), "--force"]
    print(f"[KAGGLE] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Kaggle download failed with exit code {e.returncode}. "
            f"Ensure Kaggle slug is correct and your credentials are configured."
        ) from e
    # Find the downloaded zip archives (there may be multiple)
    zips = list(download_dir.glob("*.zip"))
    if not zips:
        files = [p.name for p in download_dir.iterdir()]
        raise RuntimeError(f"No .zip archives found in {download_dir}. Files: {files}")
    return zips

def split_classification_dirs(src_root: Path, dest_root: Path, val_ratio: float, test_ratio: float, seed: int = 42) -> None:
    """
    Split a classification dataset organised by class directories into train,
    validation, and test subsets. This function follows the logic of setting
    a random seed, shuffling file names, and assigning them proportionally.
    """
    rng = np.random.RandomState(seed)
    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class directories found in {src_root}")
    # Create class subdirectories under each split
    for split in ['train', 'val', 'test']:
        for cls in class_dirs:
            (dest_root / split / cls.name).mkdir(parents=True, exist_ok=True)
    for cls in class_dirs:
        images = [p for p in cls.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
        rng.shuffle(images)
        n = len(images)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test
        for i, img_path in enumerate(images):
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
            dest = dest_root / split / cls.name / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

def split_detection_csv(csv_path: Path, images_dir: Path, dest_root: Path, val_ratio: float, test_ratio: float, seed: int = 42) -> None:
    """
    Split a detection dataset stored in a single CSV file into train, val, and
    test CSV files. The CSV should have columns: filename,xmin,ymin,xmax,ymax,label.
    All rows for a given image remain together in the same split. The images are
    copied into dest_root/images.
    """
    import csv
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    groups = {}
    for row in rows:
        fname = row[0]
        groups.setdefault(fname, []).append(row)
    rng = np.random.RandomState(seed)
    fnames = list(groups.keys())
    rng.shuffle(fnames)
    n = len(fnames)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    splits = {
        'train': fnames[:n_train],
        'val': fnames[n_train:n_train + n_val],
        'test': fnames[n_train + n_val:]
    }
    (dest_root / 'images').mkdir(parents=True, exist_ok=True)
    for split, fn_list in splits.items():
        out_csv = dest_root / f"{split}.csv"
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for fn in fn_list:
                for row in groups[fn]:
                    writer.writerow(row)
                src_img = images_dir / fn
                dst_img = dest_root / 'images' / fn
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

def split_ocr_dataset(images_dir: Path, labels_path: Path, dest_root: Path, val_ratio: float, test_ratio: float, seed: int = 42) -> None:
    """
    Split an OCR dataset described by a labels.txt file (filename<TAB>text) and
    an images directory into train, val, and test subsets. Each subset will
    reside under dest_root/{split}/images and have an associated labels.txt
    file containing only the lines for that split. Image files are copied
    accordingly.
    """
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]
    rng = np.random.RandomState(seed)
    rng.shuffle(lines)
    n = len(lines)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    splits = {
        'train': lines[:n_train],
        'val': lines[n_train:n_train + n_val],
        'test': lines[n_train + n_val:]
    }
    for split, split_lines in splits.items():
        split_img_dir = dest_root / split / 'images'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_label_file = dest_root / split / 'labels.txt'
        with open(split_label_file, 'w', encoding='utf-8') as out_f:
            for line in split_lines:
                fname, text = line.split('\t', 1)
                src_img = images_dir / fname
                dst_img = split_img_dir / fname
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                out_f.write(f"{fname}\t{text}\n")

def prepare_dataset(args: argparse.Namespace) -> None:
    """
    Prepare the dataset for training/evaluation. This function handles:
    - Downloading and extracting external archives via URL or Kaggle dataset page
    - Splitting unsplit datasets into train/val/test according to args.val_split and args.test_split
    - Creating appropriate structures for classification, detection, and OCR themes

    If args.dataset_url is a Kaggle dataset URL, the Kaggle CLI is used to
    download the dataset (requires that the Kaggle CLI and credentials are
    configured). Otherwise, the URL is treated as a direct archive (.zip or .tar).
    """
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    downloads_dir = data_root / 'downloads'
    raw_dir = data_root / 'raw'
    downloads_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # If user specified dataset URL and wants to download
    if getattr(args, 'dataset_url', None) and getattr(args, 'download', False):
        url = args.dataset_url.strip()
        print(f"[PREP] Resolving dataset from URL: {url}")
        if is_kaggle_url(url):
            slug = kaggle_slug(url)
            ensure_kaggle_cli_available()
            archives = kaggle_download_dataset(slug, downloads_dir)
            # Extract the first zip archive
            archive = archives[0]
            print(f"[PREP] Extracting Kaggle archive: {archive}")
            with zipfile.ZipFile(archive, 'r') as zf:
                zf.extractall(raw_dir)
        else:
            # Non-Kaggle: treat as direct file
            # Determine output filename
            filename = os.path.basename(url)
            if not filename:
                filename = 'data'
            dest = downloads_dir / filename
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset from {url}: {e}")
            # Extract based on extension
            suffixes = ''.join(dest.suffixes)
            if suffixes.endswith('.zip'):
                with zipfile.ZipFile(dest, 'r') as zf:
                    zf.extractall(raw_dir)
            elif suffixes.endswith('.tar.gz') or suffixes.endswith('.tgz') or suffixes.endswith('.tar'):
                mode = 'r:gz' if suffixes.endswith(('.tar.gz', '.tgz')) else 'r'
                with tarfile.open(dest, mode) as tf:
                    tf.extractall(raw_dir)
            else:
                raise RuntimeError(f"Unsupported archive format: {dest}")
    # Determine actual data root after extraction: if raw_dir has a single subdir, descend
    extracted_root = raw_dir
    subdirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        extracted_root = subdirs[0]
    # If train/val/test already exist in the extracted content, copy them over to
    # the working data_root. It's possible that only a subset of splits exist
    # (e.g. train/test or train/val). We copy whatever is provided and then
    # determine whether further splitting is needed.
    for split in ['train', 'val', 'test']:
        # Copy classification/OCR split directories if present
        src_dir = extracted_root / split
        dst_dir = data_root / split
        if src_dir.exists() and not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir)
        # Copy detection CSV split files if present
        src_csv = extracted_root / f"{split}.csv"
        dst_csv = data_root / f"{split}.csv"
        if src_csv.exists() and not dst_csv.exists():
            shutil.copy2(src_csv, dst_csv)
    # Copy images directory for detection if present
    src_images = extracted_root / 'images'
    dst_images = data_root / 'images'
    if src_images.exists() and not dst_images.exists():
        shutil.copytree(src_images, dst_images)

    # Determine how many dataset splits are already present. For classification
    # and OCR we look for split directories; for object detection we look for
    # split CSVs. If at least two splits are found, we skip splitting and use
    # the existing splits as-is. This allows datasets with train/test or
    # train/val to be used without forcing creation of a third split.
    theme = args.theme
    existing_dirs = [s for s in ['train', 'val', 'test'] if (data_root / s).exists()]
    existing_csvs = [s for s in ['train', 'val', 'test'] if (data_root / f"{s}.csv").exists()]
    if theme in ('classification', 'ocr') and len(existing_dirs) >= 2:
        print(f"[PREP] Found existing splits: {existing_dirs}; skipping splitting.")
        return
    if theme == 'object' and len(existing_csvs) >= 2:
        print(f"[PREP] Found existing splits: {existing_csvs}; skipping splitting.")
        return

    # Otherwise, need to split according to theme
    if theme == 'classification':
        # Determine classification directory: if extracted_root has no class dirs at this level, try descending
        cls_dir = extracted_root
        # Heuristic: if no subdirs or only non-image files, attempt to find deeper
        if not any(p.is_dir() for p in cls_dir.iterdir()):
            dirs = [p for p in cls_dir.iterdir() if p.is_dir()]
            if len(dirs) == 1:
                cls_dir = dirs[0]
        split_classification_dirs(
            cls_dir, data_root, args.val_split, args.test_split, seed=args.seed
        )
    elif theme == 'object':
        # Expect a single annotations CSV and images directory
        ann_candidates = list(extracted_root.glob('*.csv'))
        if len(ann_candidates) != 1:
            raise RuntimeError(
                "For object detection, provide a single CSV file with all annotations or existing train/val/test splits."
            )
        ann_csv = ann_candidates[0]
        images_dir = extracted_root / 'images'
        if not images_dir.exists():
            raise RuntimeError("Object detection dataset must contain an 'images' directory.")
        split_detection_csv(
            ann_csv, images_dir, data_root, args.val_split, args.test_split, seed=args.seed
        )
    elif theme == 'ocr':
        # Expect labels.txt and images directory if not already split
        label_files = list(extracted_root.glob('labels.txt'))
        if not label_files:
            raise RuntimeError(
                "For OCR, a labels.txt file must accompany the images directory when no train/val/test splits exist."
            )
        labels_file = label_files[0]
        images_dir = extracted_root / 'images'
        if not images_dir.exists():
            raise RuntimeError("OCR dataset must contain an 'images' directory.")
        split_ocr_dataset(
            images_dir, labels_file, data_root, args.val_split, args.test_split, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown theme: {theme}")

###########################################################################
# Plotting helpers
###########################################################################

def plot_lines(line_dict: Dict[str, List[float]], title: str, x_label: str, y_label: str, out_path: Path):
    """Plot multiple named lines on a single figure and save to out_path."""
    if not line_dict:
        return
    plt.figure(figsize=(8, 5))
    epochs = range(1, max(len(v) for v in line_dict.values()) + 1)
    for name, y in line_dict.items():
        plt.plot(epochs[:len(y)], y, label=name, linewidth=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix_matplotlib(cm: np.ndarray, class_names: List[str], out_path: Path, title: str = "Confusion Matrix"):
    """Plot a confusion matrix using Matplotlib and save to out_path."""
    if cm.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')  # use default colormap
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j])), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Models
def build_convnext_classifier(variant: str, num_classes: int, dropout: float):
    return timm.create_model(variant, pretrained=True, num_classes=num_classes, drop_rate=dropout)

def build_convnext_fpn_backbone(variant: str, out_channels=256):
    bb = timm.create_model(variant, pretrained=True, features_only=True, out_indices=(1,2,3,4))
    in_channels_list = bb.feature_info.channels()
    from torchvision.ops import FeaturePyramidNetwork

    class TimmBackboneWithFPN(nn.Module):
        def __init__(self, bb, in_chs, out_ch):
            super().__init__()
            self.body = bb
            self.fpn = FeaturePyramidNetwork(in_channels_list=in_chs, out_channels=out_ch)
            self.out_channels = out_ch

        def forward(self, x):
            feats = self.body(x)
            x = {str(i): f for i, f in enumerate(feats)}
            return self.fpn(x)

    return TimmBackboneWithFPN(bb, in_channels_list, out_channels)

def build_faster_rcnn_with_convnext(variant: str, num_classes: int):
    backbone = build_convnext_fpn_backbone(variant, out_channels=256)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes)
    return model

class OCRConvNeXtCTC(nn.Module):
    def __init__(self, variant: str, vocab_size: int, lstm_hidden=256, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = timm.create_model(variant, pretrained=True, features_only=True, out_indices=(3,))
        enc_channels = self.encoder.feature_info.channels()[-1]
        self.conv = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, 3, padding=1),
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.lstm = nn.LSTM(input_size=enc_channels, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, dropout=dropout, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(lstm_hidden*2, vocab_size)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        f = self.encoder(x)[-1] # BxCxhxw
        z = self.conv(f).mean(dim=2) # BxC x w'
        z = z.permute(2,0,1) # T x B x C
        seq, _ = self.lstm(z)
        logits = self.fc(seq) # T x B x V
        return logits

# Training / Eval loops (tqdm + AMP)
def train_one_epoch_cls(model, loader, opt, criterion, device, epoch, epochs):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    it = tqdm(loader, desc=f"Train epoch {epoch+1}/{epochs}", leave=False)
    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        with amp.autocast('cuda', dtype=torch.float16, enabled=True):
            logits = model(x)
            loss = criterion(logits, y)
        SCALER.scale(loss).backward()
        SCALER.step(opt)
        SCALER.update()
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
        it.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/max(1,total)):.3f}")
    return loss_sum / max(1, total), correct / max(1, total), None

@torch.no_grad()
def eval_cls_metrics(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    losses = 0.0; nobs = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        with amp.autocast('cuda', dtype=torch.float16, enabled=True):
            logits = model(x)
            loss = criterion(logits, y)
        losses += float(loss.item()) * x.size(0); nobs += x.size(0)
        all_preds.append(logits.argmax(1).detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    val_loss = losses / max(1, nobs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    spec = None
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp + 1e-12)

    return {"loss": val_loss, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "specificity": spec}

def train_one_epoch_det(model, loader, opt, device, epoch, epochs):
    model.train()
    loss_sum = 0.0; n = 0
    it = tqdm(loader, desc=f"Train DET {epoch+1}/{epochs}", leave=False)
    for imgs, targets in it:
        imgs = imgs.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        opt.zero_grad(set_to_none=True)
        with amp.autocast('cuda', dtype=torch.float16, enabled=True):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
        SCALER.scale(loss).backward()
        SCALER.step(opt)
        SCALER.update()
        loss_sum += float(loss.item()); n += 1
        it.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum / max(1, n), None

@torch.no_grad()
def eval_det_metrics(model, loader, device):
    model.eval()
    # TorchMetrics mAP + IoU
    map_metric = MeanAveragePrecision(iou_type="bbox")
    iou_metric = IntersectionOverUnion(iou_type="bbox") # per-image IoU mean

    loss_sum = 0.0; n = 0
    for imgs, targets in loader:
        # NOTE: for loss we need targets; for preds we need model in eval inference
        imgs = imgs.to(device, non_blocking=True)
        tgt_gpu = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        with amp.autocast('cuda', dtype=torch.float16, enabled=True):
            loss_dict = model(imgs, tgt_gpu)
            loss = sum(loss_dict.values())
        loss_sum += float(loss.item()); n += 1

        preds = model(imgs) # Eval forward for predictions
        # Convert to CPU for torchmetrics
        preds_cpu = []
        tgts_cpu  = []
        for p, t in zip(preds, targets):
            preds_cpu.append({
                "boxes": p["boxes"].detach().cpu(),
                "scores": p["scores"].detach().cpu(),
                "labels": p["labels"].detach().cpu(),
            })
            tgts_cpu.append({
                "boxes": t["boxes"].detach().cpu(),
                "labels": t["labels"].detach().cpu(),
            })
        map_metric.update(preds_cpu, tgts_cpu)
        iou_metric.update(preds_cpu, tgts_cpu)

    results = map_metric.compute() # dict with map, map_50, map_75, mar, etc.
    iou_res = iou_metric.compute() # dict with iou per class, mean on 'iou'

    val_loss = loss_sum / max(1, n)
    
    # Derive PR/F1 at IoU=0.5 (approx from torchmetrics outputs)
    mAP_05 = float(results.get("map_50", torch.tensor(0.)).item())
    mAP_50_95 = float(results.get("map", torch.tensor(0.)).item())
    
    # TorchMetrics doesn’t directly expose global P/R; we can report per-class mean recall/precision at IoU thresholds:
    # results contains e.g. "precision" / "recall" for pr curve if available; keep minimal here.
    mean_iou = float(torch.nanmean(iou_res.get("iou", torch.tensor([0.]))).item())

    return {
        "loss": val_loss,
        "mAP@0.5": mAP_05,
        "mAP@0.5:0.95": mAP_50_95,
        "mean_IoU": mean_iou
    }

def train_one_epoch_ocr(model, loader, opt, criterion_ctc, device, epoch, epochs):
    model.train()
    loss_sum = 0.0; n = 0
    it = tqdm(loader, desc=f"Train OCR {epoch+1}/{epochs}", leave=False)
    for x, target, target_lengths in it:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp.autocast('cuda', dtype=torch.float16, enabled=True):
            logits = model(x)
            T, B, V = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = criterion_ctc(logits.log_softmax(2), target, input_lengths, target_lengths)
        SCALER.scale(loss).backward()
        SCALER.step(opt)
        SCALER.update()
        loss_sum += float(loss.item()); n += 1
        it.set_postfix(ctc=f"{loss.item():.4f}")
    return loss_sum / max(1, n), None

@torch.no_grad()
def eval_ocr_metrics(model, loader, criterion_ctc, device):
    model.eval()
    loss_sum = 0.0; n = 0
    preds_all: List[str] = []
    gts_all:   List[str] = []
    encoder = loader.dataset.encoder

    for x, target, target_lengths in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        with amp.autocast('cuda', dtype=torch.float16, enabled=True):
            logits = model(x)
            T, B, V = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = criterion_ctc(logits.log_softmax(2), target, input_lengths, target_lengths)
        loss_sum += float(loss.item()); n += 1

        # Decode greedy
        pred_texts = encoder.decode_greedy(logits)
        # Reconstruct GT strings
        offset = 0
        inv = encoder.idx2ch
        gt_texts = []
        for L in target_lengths.cpu().tolist():
            seq = target[offset:offset+L].detach().cpu().tolist()
            gt_texts.append("".join(inv.get(i, "") for i in seq))
            offset += L

        preds_all.extend(pred_texts)
        gts_all.extend(gt_texts)

    val_loss = loss_sum / max(1, n)
    
    # CER/WER
    cer_val = cer(gts_all, preds_all) if len(gts_all) else 1.0
    wer_val = wer(gts_all, preds_all) if len(gts_all) else 1.0

    # Char-level precision/recall/F1 (tokenize to single chars)
    y_true_chars = list("".join(gts_all))
    y_pred_chars = list("".join(preds_all))
    
    # Align length for macro metrics — compute on set of seen characters
    uniq_chars = sorted(list(set(y_true_chars + y_pred_chars)))
    char_to_idx = {ch: i for i, ch in enumerate(uniq_chars)}
    y_true_idx = [char_to_idx[ch] for ch in y_true_chars] if y_true_chars else []
    y_pred_idx = [char_to_idx[ch] for ch in y_pred_chars] if y_pred_chars else []
    
    # For different lengths, pad shorter with a special value to avoid errors (and ignore in metrics)
    # Simpler: compute macro-PRF using confusion on overlapping length
    L = min(len(y_true_idx), len(y_pred_idx))
    if L == 0:
        prec = rec = f1v = 0.0
    else:
        yt = np.array(y_true_idx[:L])
        yp = np.array(y_pred_idx[:L])
        prec = precision_score(yt, yp, average="macro", zero_division=0)
        rec  = recall_score(yt, yp, average="macro", zero_division=0)
        f1v  = f1_score(yt, yp, average="macro", zero_division=0)

    return {
        "loss": val_loss,
        "CER": cer_val,
        "WER": wer_val,
        "precision_char": float(prec),
        "recall_char": float(rec),
        "f1_char": float(f1v),
    }

# Optuna objective
def objective_factory(args, theme: str):
    seed_everything(args.seed)
    device = device_auto()
    out_dir = Path(args.out_dir) / theme
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizers = ["adamw", "sgd"]

    def objective(trial: optuna.Trial):
        variant = MANUAL_CONVNEXT_VARIANT
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        opt_name = trial.suggest_categorical("optimizer", optimizers)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])

        if theme == "classification":
            # Set up transforms
            t_train = ClassificationTransform(size=args.img_size, train=True)
            t_val   = ClassificationTransform(size=args.img_size, train=False)
            # Load training dataset and infer class mapping
            ds_train = ClassificationDataset(args.data_root, "train", t_train)
            class_to_idx = ds_train.class_to_idx
            num_classes = len(class_to_idx)
            if num_classes < 2:
                raise RuntimeError("Need at least 2 classes.")
            # Determine which split to use for validation: prefer 'val', else 'test'
            from pathlib import Path as _P
            val_split_name = "val" if (_P(args.data_root) / "val").exists() else "test"
            # Load validation dataset using the same class mapping
            ds_val   = ClassificationDataset(args.data_root, val_split_name, t_val, class_to_idx=class_to_idx)
            # Data loaders
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
            dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

            # Model and optimizer
            model = build_convnext_classifier(variant, num_classes, dropout).to(device)
            opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                   if opt_name == "adamw"
                   else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True))
            criterion = nn.CrossEntropyLoss()

            # Metric history containers
            train_losses: List[float] = []
            train_accs:   List[float] = []
            val_losses:   List[float] = []
            val_accs:     List[float] = []
            val_precisions: List[float] = []
            val_recalls:    List[float] = []
            val_f1s:        List[float] = []

            best_acc = -1.0
            best_path = out_dir / "best_model.pth"
            # Training loop
            for epoch in range(args.epochs):
                # Train one epoch
                tr_loss, tr_acc, _ = train_one_epoch_cls(model, dl_train, opt, criterion, device, epoch, args.epochs)
                # Validate
                mets = eval_cls_metrics(model, dl_val, criterion, device)
                # Append metrics
                train_losses.append(tr_loss)
                train_accs.append(tr_acc)
                val_losses.append(mets['loss'])
                val_accs.append(mets['acc'])
                val_precisions.append(mets['precision'])
                val_recalls.append(mets['recall'])
                val_f1s.append(mets['f1'])
                # Print a concise summary per epoch with train and val metrics
                print(f"Epoch {epoch+1}/{args.epochs}: "
                      f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
                      f"val_loss={mets['loss']:.4f} val_acc={mets['acc']:.3f} "
                      f"prec={mets['precision']:.3f} rec={mets['recall']:.3f} f1={mets['f1']:.3f}"
                      + (f" spec={mets['specificity']:.3f}" if mets['specificity'] is not None else ""))
                # Report to Optuna (we maximize accuracy, so minimize 1-acc)
                trial.report(1.0 - mets['acc'], step=epoch)
                # Update best model based on validation accuracy
                if mets['acc'] > best_acc:
                    best_acc = mets['acc']
                    # Save model checkpoint along with metadata and class mapping
                    torch.save({
                        "model": model.state_dict(),
                        "variant": variant,
                        "num_classes": num_classes,
                        "classes": list(class_to_idx.keys())
                    }, best_path)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            # After all epochs, save training history and plots
            # Save metric arrays as .txt for easy reading and as .npy for reproducibility
            np.savetxt(out_dir / f"train_losses_trial{trial.number}.txt", np.array(train_losses), fmt="%.6f")
            np.savetxt(out_dir / f"train_accs_trial{trial.number}.txt",   np.array(train_accs),   fmt="%.6f")
            np.savetxt(out_dir / f"val_losses_trial{trial.number}.txt",   np.array(val_losses),   fmt="%.6f")
            np.savetxt(out_dir / f"val_accs_trial{trial.number}.txt",     np.array(val_accs),     fmt="%.6f")
            # Plot training vs validation loss
            plot_lines(
                {
                    "Train Loss": train_losses,
                    "Val Loss": val_losses,
                },
                title="Training vs Validation Loss",
                x_label="Epoch",
                y_label="Loss",
                out_path=out_dir / f"loss_curve_trial{trial.number}.png"
            )
            # Plot training vs validation accuracy
            plot_lines(
                {
                    "Train Accuracy": train_accs,
                    "Val Accuracy": val_accs,
                },
                title="Training vs Validation Accuracy",
                x_label="Epoch",
                y_label="Accuracy",
                out_path=out_dir / f"acc_curve_trial{trial.number}.png"
            )
            # Plot confusion matrix of last validation epoch
            # Compute confusion matrix on the full validation set using current model
            with torch.no_grad():
                all_preds_cm: List[np.ndarray] = []
                all_labels_cm: List[np.ndarray] = []
                for x_val, y_val in dl_val:
                    x_val = x_val.to(device, non_blocking=True)
                    y_val = y_val.to(device, non_blocking=True)
                    logits_val = model(x_val)
                    all_preds_cm.append(logits_val.argmax(1).detach().cpu().numpy())
                    all_labels_cm.append(y_val.detach().cpu().numpy())
                if all_labels_cm:
                    y_true_cm = np.concatenate(all_labels_cm)
                    y_pred_cm = np.concatenate(all_preds_cm)
                    cm = confusion_matrix(y_true_cm, y_pred_cm)
                    plot_confusion_matrix_matplotlib(
                        cm,
                        list(class_to_idx.keys()),
                        out_path=out_dir / f"val_confusion_matrix_trial{trial.number}.png",
                        title="Validation Confusion Matrix"
                    )
            # Write a summary text for this trial
            summary_lines = []
            summary_lines.append(f"Trial {trial.number} summary:\n")
            summary_lines.append(f"Best validation accuracy: {best_acc:.4f}\n")
            summary_lines.append(f"Hyperparameters: lr={lr:.6f}, weight_decay={wd:.6f}, dropout={dropout:.3f}, optimizer={opt_name}, batch_size={batch_size}\n")
            summary_lines.append("Epoch-wise metrics:\n")
            for i in range(len(train_losses)):
                summary_lines.append(f"Epoch {i+1}: train_loss={train_losses[i]:.4f}, train_acc={train_accs[i]:.4f}, "
                                     f"val_loss={val_losses[i]:.4f}, val_acc={val_accs[i]:.4f}, "
                                     f"val_prec={val_precisions[i]:.4f}, val_rec={val_recalls[i]:.4f}, val_f1={val_f1s[i]:.4f}\n")
            with open(out_dir / f"trial_{trial.number}_summary.txt", "w", encoding="utf-8") as sf:
                sf.writelines(summary_lines)
            # Return objective value: minimize 1 - best accuracy
            return 1.0 - float(best_acc)

        elif theme == "object":
            # Object detection hyperparameter experiment. We will log training and
            # validation losses and metrics (mAP and IoU) similar to the classification
            # pipeline. If two detection splits (train/val) are available, we do not
            # create a third split and train/validate on those.
            t_train = ObjectDetectionTransform(size=args.det_size, train=True)
            t_val   = ObjectDetectionTransform(size=args.det_size, train=False)
            ds_train = ObjectDetectionDataset(args.data_root, "train", t_train)
            # Determine which split to use for validation: prefer 'val.csv', else 'test.csv'
            from pathlib import Path as _P
            val_split_name_det = "val" if (_P(args.data_root) / "val.csv").exists() else "test"
            ds_val   = ObjectDetectionDataset(args.data_root, val_split_name_det, t_val)
            num_classes = len(ds_train.classes) + 1
            if num_classes < 2:
                raise RuntimeError("Need at least 1 foreground class.")
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, collate_fn=detection_collate)
            dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True, collate_fn=detection_collate)

            model = build_faster_rcnn_with_convnext(variant, num_classes=num_classes).to(device)
            opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                   if opt_name == "adamw"
                   else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True))

            # History containers for logging
            train_losses: List[float] = []
            val_losses:   List[float] = []
            val_map05:    List[float] = []
            val_map5095:  List[float] = []
            val_iou:      List[float] = []

            best_map = -1.0
            best_path = out_dir / "best_model.pth"
            for epoch in range(args.epochs):
                # Train one epoch
                tr_loss, _ = train_one_epoch_det(model, dl_train, opt, device, epoch, args.epochs)
                # Validate
                mets = eval_det_metrics(model, dl_val, device)
                # Record metrics
                train_losses.append(tr_loss)
                val_losses.append(mets['loss'])
                val_map05.append(mets['mAP@0.5'])
                val_map5095.append(mets['mAP@0.5:0.95'])
                val_iou.append(mets['mean_IoU'])
                # Print a concise summary per epoch with train and validation metrics
                print(
                    f"Epoch {epoch+1}/{args.epochs}: train_loss={tr_loss:.4f} | "
                    f"val_loss={mets['loss']:.4f} mAP@0.5={mets['mAP@0.5']:.3f} "
                    f"mAP@0.5:0.95={mets['mAP@0.5:0.95']:.3f} meanIoU={mets['mean_IoU']:.3f}"
                )
                # Report to Optuna (maximize mAP@0.5:0.95 by minimizing its negative)
                trial.report(-mets["mAP@0.5:0.95"], step=epoch)
                # Update best model
                if mets["mAP@0.5:0.95"] > best_map:
                    best_map = mets["mAP@0.5:0.95"]
                    torch.save({
                        "model": model.state_dict(),
                        "variant": variant,
                        "classes": ds_train.classes
                    }, best_path)
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # After all epochs, save history and plots for this trial
            np.savetxt(out_dir / f"det_train_losses_trial{trial.number}.txt", np.array(train_losses), fmt="%.6f")
            np.savetxt(out_dir / f"det_val_losses_trial{trial.number}.txt",   np.array(val_losses),   fmt="%.6f")
            np.savetxt(out_dir / f"det_val_map50_trial{trial.number}.txt",    np.array(val_map05),    fmt="%.6f")
            np.savetxt(out_dir / f"det_val_map50_95_trial{trial.number}.txt", np.array(val_map5095),  fmt="%.6f")
            np.savetxt(out_dir / f"det_val_mean_iou_trial{trial.number}.txt", np.array(val_iou),      fmt="%.6f")
            # Plot training vs validation loss
            plot_lines(
                {"Train Loss": train_losses, "Val Loss": val_losses},
                title="Detection Train vs Val Loss",
                x_label="Epoch",
                y_label="Loss",
                out_path=out_dir / f"det_loss_curve_trial{trial.number}.png"
            )
            # Plot validation mAP/IoU metrics
            plot_lines(
                {
                    "mAP@0.5": val_map05,
                    "mAP@0.5:0.95": val_map5095,
                    "Mean IoU": val_iou,
                },
                title="Detection Validation Metrics",
                x_label="Epoch",
                y_label="Metric",
                out_path=out_dir / f"det_metrics_curve_trial{trial.number}.png"
            )
            # Write summary text for this trial
            summary_lines = []
            summary_lines.append(f"Trial {trial.number} summary:\n")
            summary_lines.append(f"Best mAP@0.5:0.95: {best_map:.4f}\n")
            summary_lines.append(
                f"Hyperparameters: lr={lr:.6f}, weight_decay={wd:.6f}, "
                f"dropout={dropout:.3f}, optimizer={opt_name}, batch_size={batch_size}\n"
            )
            summary_lines.append("Epoch-wise metrics:\n")
            for i in range(len(train_losses)):
                summary_lines.append(
                    f"Epoch {i+1}: train_loss={train_losses[i]:.4f}, "
                    f"val_loss={val_losses[i]:.4f}, mAP@0.5={val_map05[i]:.4f}, "
                    f"mAP@0.5:0.95={val_map5095[i]:.4f}, mean_IoU={val_iou[i]:.4f}\n"
                )
            with open(out_dir / f"trial_{trial.number}_summary.txt", "w", encoding="utf-8") as sf:
                sf.writelines(summary_lines)

            # Return objective value: minimize negative best mAP (maximize mAP)
            return -float(best_map)

        elif theme == "ocr":
            # OCR hyperparameter experiment. Similar to classification, we log
            # training and validation losses and error metrics and save plots.
            t_train = OCRTransform(img_h=args.ocr_h, img_w=args.ocr_w, train=True)
            t_val   = OCRTransform(img_h=args.ocr_h, img_w=args.ocr_w, train=False)
            ds_train = OCRDataset(args.data_root, "train", t_train, args.charset)
            # Determine which split to use for validation: prefer 'val', else 'test'
            from pathlib import Path as _P
            val_split_name_ocr = "val" if (_P(args.data_root) / "val").exists() else "test"
            ds_val   = OCRDataset(args.data_root, val_split_name_ocr, t_val, args.charset)
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, collate_fn=ocr_collate)
            dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True, collate_fn=ocr_collate)

            # Suggest LSTM hyperparameters from trial
            lstm_hidden = trial.suggest_categorical("lstm_hidden", [192, 256, 384])
            lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2, 3])

            model = OCRConvNeXtCTC(
                variant=variant,
                vocab_size=ds_train.encoder.vocab_size,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=dropout
            ).to(device)
            opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                   if opt_name == "adamw"
                   else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True))
            ctc = nn.CTCLoss(blank=0, zero_infinity=True)

            # History containers for logging
            train_losses: List[float] = []
            val_losses:   List[float] = []
            val_cers:     List[float] = []
            val_wers:     List[float] = []
            val_prec_chars: List[float] = []
            val_rec_chars:  List[float] = []
            val_f1_chars:   List[float] = []

            best_cer = 1e9
            best_path = out_dir / "best_model.pth"
            for epoch in range(args.epochs):
                # Train one epoch
                tr_loss, _ = train_one_epoch_ocr(model, dl_train, opt, ctc, device, epoch, args.epochs)
                # Validate
                mets = eval_ocr_metrics(model, dl_val, ctc, device)
                # Record metrics
                train_losses.append(tr_loss)
                val_losses.append(mets['loss'])
                val_cers.append(mets['CER'])
                val_wers.append(mets['WER'])
                val_prec_chars.append(mets['precision_char'])
                val_rec_chars.append(mets['recall_char'])
                val_f1_chars.append(mets['f1_char'])
                # Print a concise summary per epoch with train and validation metrics
                print(
                    f"Epoch {epoch+1}/{args.epochs}: train_loss={tr_loss:.4f} | "
                    f"val_ctc={mets['loss']:.4f} CER={mets['CER']:.3f} WER={mets['WER']:.3f} "
                    f"prec_char={mets['precision_char']:.3f} rec_char={mets['recall_char']:.3f} "
                    f"f1_char={mets['f1_char']:.3f}"
                )
                # Report to Optuna (maximize negative CER to minimize CER)
                trial.report(-mets["CER"], step=epoch)
                # Update best model
                if mets["CER"] < best_cer:
                    best_cer = mets["CER"]
                    torch.save({
                        "model": model.state_dict(),
                        "variant": variant,
                        "charset": args.charset,
                        "lstm_hidden": lstm_hidden,
                        "lstm_layers": lstm_layers,
                        "dropout": dropout
                    }, best_path)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # After all epochs, save metric history and plots
            np.savetxt(out_dir / f"ocr_train_losses_trial{trial.number}.txt", np.array(train_losses), fmt="%.6f")
            np.savetxt(out_dir / f"ocr_val_losses_trial{trial.number}.txt",   np.array(val_losses),   fmt="%.6f")
            np.savetxt(out_dir / f"ocr_val_cers_trial{trial.number}.txt",    np.array(val_cers),     fmt="%.6f")
            np.savetxt(out_dir / f"ocr_val_wers_trial{trial.number}.txt",    np.array(val_wers),     fmt="%.6f")
            np.savetxt(out_dir / f"ocr_val_prec_chars_trial{trial.number}.txt", np.array(val_prec_chars), fmt="%.6f")
            np.savetxt(out_dir / f"ocr_val_rec_chars_trial{trial.number}.txt",  np.array(val_rec_chars),  fmt="%.6f")
            np.savetxt(out_dir / f"ocr_val_f1_chars_trial{trial.number}.txt",   np.array(val_f1_chars),   fmt="%.6f")
            # Plot training vs validation loss
            plot_lines(
                {"Train Loss": train_losses, "Val Loss": val_losses},
                title="OCR Train vs Val CTC Loss",
                x_label="Epoch",
                y_label="CTC Loss",
                out_path=out_dir / f"ocr_loss_curve_trial{trial.number}.png"
            )
            # Plot error rates
            plot_lines(
                {"CER": val_cers, "WER": val_wers},
                title="OCR Error Rates",
                x_label="Epoch",
                y_label="Error",
                out_path=out_dir / f"ocr_error_curve_trial{trial.number}.png"
            )
            # Plot character-level precision/recall/F1
            plot_lines(
                {
                    "Precision": val_prec_chars,
                    "Recall": val_rec_chars,
                    "F1": val_f1_chars,
                },
                title="OCR Character-level Metrics",
                x_label="Epoch",
                y_label="Score",
                out_path=out_dir / f"ocr_char_metrics_curve_trial{trial.number}.png"
            )
            # Write summary text for this trial
            summary_lines = []
            summary_lines.append(f"Trial {trial.number} summary:\n")
            summary_lines.append(f"Best CER: {best_cer:.4f}\n")
            summary_lines.append(
                f"Hyperparameters: lr={lr:.6f}, weight_decay={wd:.6f}, dropout={dropout:.3f}, "
                f"optimizer={opt_name}, batch_size={batch_size}, lstm_hidden={lstm_hidden}, lstm_layers={lstm_layers}\n"
            )
            summary_lines.append("Epoch-wise metrics:\n")
            for i in range(len(train_losses)):
                summary_lines.append(
                    f"Epoch {i+1}: train_loss={train_losses[i]:.4f}, val_loss={val_losses[i]:.4f}, "
                    f"CER={val_cers[i]:.4f}, WER={val_wers[i]:.4f}, "
                    f"prec_char={val_prec_chars[i]:.4f}, rec_char={val_rec_chars[i]:.4f}, "
                    f"f1_char={val_f1_chars[i]:.4f}\n"
                )
            with open(out_dir / f"trial_{trial.number}_summary.txt", "w", encoding="utf-8") as sf:
                sf.writelines(summary_lines)

            # Return best CER as objective (smaller is better)
            return float(best_cer)

        else:
            raise ValueError(f"Unknown theme: {theme}")

    return objective

# Test-time evaluation (loads best ckpt)
@torch.no_grad()
def evaluate_on_test(args, theme: str):
    device = device_auto()
    out_dir = Path(args.out_dir) / theme
    ckpt = out_dir / "best_model.pth"
    if not ckpt.exists():
        print("[TEST] No best_model.pth found, skipping test eval.")
        return

    print("[TEST] Loading:", ckpt)
    sd = torch.load(ckpt, map_location=device)

    if theme == "classification":
        t_test = ClassificationTransform(size=args.img_size, train=False)
        # Use saved class mapping if available
        class_names = sd.get("classes")
        if class_names is not None:
            class_to_idx = {c: i for i, c in enumerate(class_names)}
            ds_test = ClassificationDataset(args.data_root, "test", t_test, class_to_idx=class_to_idx)
        else:
            ds_test = ClassificationDataset(args.data_root, "test", t_test)
            class_names = list(ds_test.class_to_idx.keys())
        dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=args.workers, pin_memory=True)
        variant = sd["variant"]; num_classes = sd["num_classes"]
        model = build_convnext_classifier(variant, num_classes, dropout=0.0).to(device)
        model.load_state_dict(sd["model"])
        criterion = nn.CrossEntropyLoss()
        mets = eval_cls_metrics(model, dl_test, criterion, device)
        print(f"[TEST] loss={mets['loss']:.4f} acc={mets['acc']:.3f} "
              f"prec={mets['precision']:.3f} rec={mets['recall']:.3f} f1={mets['f1']:.3f}"
              + (f" spec={mets['specificity']:.3f}" if mets['specificity'] is not None else ""))
        # Save metrics to JSON and plain text
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(mets, f, indent=2)
        # Save as plain text summary
        with open(out_dir / "test_metrics.txt", "w", encoding="utf-8") as tf:
            for k, v in mets.items():
                tf.write(f"{k}: {v}\n")
        # Plot confusion matrix on test set if available
        # Compute y_true, y_pred manually to get confusion matrix
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        for x_test, y_test in dl_test:
            x_test = x_test.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)
            logits = model(x_test)
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_labels.append(y_test.detach().cpu().numpy())
        if all_labels:
            y_true_test = np.concatenate(all_labels)
            y_pred_test = np.concatenate(all_preds)
            cm = confusion_matrix(y_true_test, y_pred_test)
            plot_confusion_matrix_matplotlib(
                cm,
                class_names,
                out_path=out_dir / "test_confusion_matrix.png",
                title="Test Confusion Matrix"
            )

    elif theme == "object":
        t_test = ObjectDetectionTransform(size=args.det_size, train=False)
        ds_test = ObjectDetectionDataset(args.data_root, "test", t_test)
        dl_test = DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=args.workers,
                             pin_memory=True, collate_fn=detection_collate)
        variant = sd["variant"]; classes = sd["classes"]
        num_classes = len(classes) + 1
        model = build_faster_rcnn_with_convnext(variant, num_classes=num_classes).to(device)
        model.load_state_dict(sd["model"])
        mets = eval_det_metrics(model, dl_test, device)
        print(f"[TEST DET] loss={mets['loss']:.4f} mAP@0.5={mets['mAP@0.5']:.3f} "
              f"mAP@0.5:0.95={mets['mAP@0.5:0.95']:.3f} meanIoU={mets['mean_IoU']:.3f}")
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(mets, f, indent=2)
        # Save plain text summary
        with open(out_dir / "test_metrics.txt", "w", encoding="utf-8") as tf:
            for k, v in mets.items():
                tf.write(f"{k}: {v}\n")

    elif theme == "ocr":
        # Build test loader
        t_test = OCRTransform(img_h=args.ocr_h, img_w=args.ocr_w, train=False)
        # Determine charset from checkpoint or args
        charset = sd.get("charset", args.charset)
        ds_test = OCRDataset(args.data_root, "test", t_test, charset)
        dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=args.workers,
                             pin_memory=True, collate_fn=ocr_collate)
        variant = sd.get("variant", MANUAL_CONVNEXT_VARIANT)
        vocab_size = ds_test.encoder.vocab_size
        # Determine LSTM hyperparams from checkpoint or best_params.json; fallback to defaults
        lstm_hidden = sd.get("lstm_hidden")
        lstm_layers = sd.get("lstm_layers")
        dropout = sd.get("dropout", 0.0)
        if lstm_hidden is None or lstm_layers is None:
            # Try reading from best_params.json if available
            try:
                params_path = out_dir / "best_params.json"
                if params_path.exists():
                    with open(params_path, "r", encoding="utf-8") as pf:
                        params = json.load(pf)
                    lstm_hidden = int(params.get("lstm_hidden", 256))
                    lstm_layers = int(params.get("lstm_layers", 2))
                    dropout = float(params.get("dropout", dropout))
            except Exception:
                pass
        if lstm_hidden is None:
            lstm_hidden = 256
        if lstm_layers is None:
            lstm_layers = 2
        # Build model with the discovered hyperparameters
        model = OCRConvNeXtCTC(
            variant=variant,
            vocab_size=vocab_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout
        ).to(device)
        # Load weights
        model.load_state_dict(sd["model"])
        ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        mets = eval_ocr_metrics(model, dl_test, ctc, device)
        print(
            f"[TEST OCR] ctc={mets['loss']:.4f} CER={mets['CER']:.3f} WER={mets['WER']:.3f} "
            f"prec_char={mets['precision_char']:.3f} rec_char={mets['recall_char']:.3f} "
            f"f1_char={mets['f1_char']:.3f}"
        )
        # Save metrics
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(mets, f, indent=2)
        with open(out_dir / "test_metrics.txt", "w", encoding="utf-8") as tf:
            for k, v in mets.items():
                tf.write(f"{k}: {v}\n")

    else:
        print("[TEST] Unknown theme; skipping.")

# CLI / Main
def parse_args():
    """Parse command line arguments including dataset handling options."""
    p = argparse.ArgumentParser(description="ConvNeXt + Optuna (train/val/test) with metrics")
    # Core paths and training parameters
    p.add_argument("--data_root", type=str, required=True, help="Dataset root path or working directory")
    p.add_argument("--out_dir", type=str, default="./artifacts", help="Output directory for all artifacts")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers (0 for Windows compatibility)")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs for each trial")
    p.add_argument("--n_trials", type=int, default=1, help="Number of Optuna hyperparameter trials")
    p.add_argument("--study_name", type=str, default=None, help="Optional Optuna study name (to resume previous studies)")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URI (e.g. sqlite:///study.db)")
    # Dataset ingestion options
    p.add_argument("--dataset_url", type=str, default=None,
                   help="URL to a compressed dataset archive (.zip/.tar) or Kaggle dataset page")
    p.add_argument("--download", action="store_true",
                   help="If set, automatically download and extract the dataset from --dataset_url")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of data to allocate for validation when splitting (0 < val_split < 1)")
    p.add_argument("--test_split", type=float, default=0.1,
                   help="Fraction of data to allocate for testing when splitting (0 < test_split < 1)")
    # Classification specific
    p.add_argument("--img_size", type=int, default=224, help="Input image size for classification models")
    # Detection specific
    p.add_argument("--det_size", type=int, default=640, help="Input image size for object detection models")
    # OCR specific
    p.add_argument("--ocr_h", type=int, default=32, help="OCR image height for ConvNeXt-CTC")
    p.add_argument("--ocr_w", type=int, default=256, help="OCR image width for ConvNeXt-CTC")
    p.add_argument("--charset", type=str, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                   help="Character set used for OCR tasks")
    # Pipeline theme
    default_theme = os.environ.get("IF_PIPELINE_THEME", "classification").lower()
    p.add_argument("--theme", type=str, default=default_theme,
                   choices=["classification", "object", "ocr"],
                   help="Pipeline theme: classification, object (detection), or ocr (text recognition)")
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    theme = args.theme
    device = device_auto()
    print(f"[INFO] Theme: {theme}")
    print(f"[INFO] Using manual ConvNeXt variant: {MANUAL_CONVNEXT_VARIANT}")
    print(f"[INFO] CUDA: {torch.cuda.is_available()} | device: {device}")

    # Prepare dataset (download/extract/split) if requested
    try:
        prepare_dataset(args)
    except Exception as prep_err:
        print(f"[ERROR] Dataset preparation failed: {prep_err}")
        return

    objective = objective_factory(args, theme)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1)

    if args.study_name or args.storage:
        if not args.study_name:
            args.study_name = f"{theme}_study"
        if not args.storage:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            args.storage = f"sqlite:///{Path(args.out_dir) / theme / 'study.db'}"
        study = optuna.create_study(direction="minimize", study_name=args.study_name,
                                    storage=args.storage, load_if_exists=True, pruner=pruner)
    else:
        study = optuna.create_study(direction="minimize", pruner=pruner)

    print("[INFO] Starting Optuna optimization...")
    wall = time.time()
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    total = time.time() - wall

    print(f"[INFO] Best trial: {study.best_trial.number}")
    print(f"[INFO] Best value (lower is better): {study.best_value:.6f}")
    print(f"[INFO] Best params: {study.best_params}")
    print(f"[INFO] Total HPO time: {total/60:.1f}m")

    # Save best params
    out_dir = Path(args.out_dir) / theme
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    # Evaluate on test split with best checkpoint
    evaluate_on_test(args, theme)
    print(f"[OK] Done. Artifacts and test metrics in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()