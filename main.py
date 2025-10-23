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
    def __init__(self, root: str, split: str, transform: ClassificationTransform):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        classes = sorted([p.name for p in (self.root / split).iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for imgp in (self.root / split / c).glob("*.*"):
                if imgp.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.samples.append((str(imgp), self.class_to_idx[c]))

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
            t_train = ClassificationTransform(size=args.img_size, train=True)
            t_val   = ClassificationTransform(size=args.img_size, train=False)
            ds_train = ClassificationDataset(args.data_root, "train", t_train)
            ds_val   = ClassificationDataset(args.data_root, "val", t_val)
            num_classes = len(ds_train.class_to_idx)
            if num_classes < 2:
                raise RuntimeError("Need at least 2 classes.")
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
            dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

            model = build_convnext_classifier(variant, num_classes, dropout).to(device)
            opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                   if opt_name == "adamw"
                   else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True))
            criterion = nn.CrossEntropyLoss()

            best_acc = -1.0
            best_path = out_dir / "best_model.pth"
            for epoch in range(args.epochs):
                tr_loss, tr_acc, _ = train_one_epoch_cls(model, dl_train, opt, criterion, device, epoch, args.epochs)
                mets = eval_cls_metrics(model, dl_val, criterion, device)
                print(f"[Val] loss={mets['loss']:.4f} acc={mets['acc']:.3f} "
                      f"prec={mets['precision']:.3f} rec={mets['recall']:.3f} f1={mets['f1']:.3f}"
                      + (f" spec={mets['specificity']:.3f}" if mets['specificity'] is not None else ""))
                trial.report(mets["acc"], step=epoch)
                if mets["acc"] > best_acc:
                    best_acc = mets["acc"]
                    torch.save({"model": model.state_dict(), "variant": variant, "num_classes": num_classes},
                               best_path)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 1.0 - float(best_acc)

        elif theme == "object":
            t_train = ObjectDetectionTransform(size=args.det_size, train=True)
            t_val   = ObjectDetectionTransform(size=args.det_size, train=False)
            ds_train = ObjectDetectionDataset(args.data_root, "train", t_train)
            ds_val   = ObjectDetectionDataset(args.data_root, "val", t_val)
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

            best_map = -1.0
            best_path = out_dir / "best_model.pth"
            for epoch in range(args.epochs):
                tr_loss, _ = train_one_epoch_det(model, dl_train, opt, device, epoch, args.epochs)
                mets = eval_det_metrics(model, dl_val, device)
                print(f"[Val DET] loss={mets['loss']:.4f} mAP@0.5={mets['mAP@0.5']:.3f} "
                      f"mAP@0.5:0.95={mets['mAP@0.5:0.95']:.3f} meanIoU={mets['mean_IoU']:.3f}")
                trial.report(mets["mAP@0.5:0.95"], step=epoch)
                if mets["mAP@0.5:0.95"] > best_map:
                    best_map = mets["mAP@0.5:0.95"]
                    torch.save({"model": model.state_dict(), "variant": variant, "classes": ds_train.classes},
                               best_path)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 1.0 - float(best_map)

        elif theme == "ocr":
            t_train = OCRTransform(img_h=args.ocr_h, img_w=args.ocr_w, train=True)
            t_val   = OCRTransform(img_h=args.ocr_h, img_w=args.ocr_w, train=False)
            ds_train = OCRDataset(args.data_root, "train", t_train, args.charset)
            ds_val   = OCRDataset(args.data_root, "val", t_val, args.charset)
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, collate_fn=ocr_collate)
            dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True, collate_fn=ocr_collate)

            model = OCRConvNeXtCTC(variant=variant, vocab_size=ds_train.encoder.vocab_size,
                                   lstm_hidden=trial.suggest_categorical("lstm_hidden", [192,256,384]),
                                   lstm_layers=trial.suggest_categorical("lstm_layers", [1,2,3]),
                                   dropout=dropout).to(device)
            opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                   if opt_name == "adamw"
                   else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True))
            ctc = nn.CTCLoss(blank=0, zero_infinity=True)

            best_cer = 1e9
            best_path = out_dir / "best_model.pth"
            for epoch in range(args.epochs):
                tr_loss, _ = train_one_epoch_ocr(model, dl_train, opt, ctc, device, epoch, args.epochs)
                mets = eval_ocr_metrics(model, dl_val, ctc, device)
                print(f"[Val OCR] ctc={mets['loss']:.4f} CER={mets['CER']:.3f} WER={mets['WER']:.3f} "
                      f"prec_char={mets['precision_char']:.3f} rec_char={mets['recall_char']:.3f} "
                      f"f1_char={mets['f1_char']:.3f}")
                trial.report(-mets["CER"], step=epoch)  # minimize CER -> report negative CER to maximize
                if mets["CER"] < best_cer:
                    best_cer = mets["CER"]
                    torch.save({"model": model.state_dict(), "variant": variant, "charset": args.charset},
                               best_path)
                if trial.should_prune():
                    raise optuna.TrialPruned()
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
        ds_test = ClassificationDataset(args.data_root, "test", t_test)
        dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=args.workers, pin_memory=True)
        variant = sd["variant"]; num_classes = sd["num_classes"]
        model = build_convnext_classifier(variant, num_classes, dropout=0.0).to(device)
        model.load_state_dict(sd["model"])
        criterion = nn.CrossEntropyLoss()
        mets = eval_cls_metrics(model, dl_test, criterion, device)
        print(f"[TEST] loss={mets['loss']:.4f} acc={mets['acc']:.3f} "
              f"prec={mets['precision']:.3f} rec={mets['recall']:.3f} f1={mets['f1']:.3f}"
              + (f" spec={mets['specificity']:.3f}" if mets['specificity'] is not None else ""))
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(mets, f, indent=2)

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
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(mets, f, indent=2)

    elif theme == "ocr":
        t_test = OCRTransform(img_h=args.ocr_h, img_w=args.ocr_w, train=False)
        ds_test = OCRDataset(args.data_root, "test", t_test, sd.get("charset", args.charset))
        dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=args.workers,
                             pin_memory=True, collate_fn=ocr_collate)
        variant = sd["variant"]; vocab = ds_test.encoder.vocab_size
        model = OCRConvNeXtCTC(variant=variant, vocab_size=vocab).to(device)
        model.load_state_dict(sd["model"])
        ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        mets = eval_ocr_metrics(model, dl_test, ctc, device)
        print(f"[TEST OCR] ctc={mets['loss']:.4f} CER={mets['CER']:.3f} WER={mets['WER']:.3f} "
              f"prec_char={mets['precision_char']:.3f} rec_char={mets['recall_char']:.3f} "
              f"f1_char={mets['f1_char']:.3f}")
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(mets, f, indent=2)

    else:
        print("[TEST] Unknown theme; skipping.")

# CLI / Main
def parse_args():
    p = argparse.ArgumentParser(description="ConvNeXt + Optuna (train/val/test) with metrics")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    p.add_argument("--out_dir", type=str, default="./artifacts", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0) # Windows-safe
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--n_trials", type=int, default=1)
    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None)
    # classification
    p.add_argument("--img_size", type=int, default=224)
    # detection
    p.add_argument("--det_size", type=int, default=640)
    # ocr
    p.add_argument("--ocr_h", type=int, default=32)
    p.add_argument("--ocr_w", type=int, default=256)
    p.add_argument("--charset", type=str, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    default_theme = os.environ.get("IF_PIPELINE_THEME", "classification").lower()
    p.add_argument("--theme", type=str, default=default_theme, choices=["classification", "object", "ocr"])
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    theme = args.theme
    device = device_auto()
    print(f"[INFO] Theme: {theme}")
    print(f"[INFO] Using manual ConvNeXt variant: {MANUAL_CONVNEXT_VARIANT}")
    print(f"[INFO] CUDA: {torch.cuda.is_available()} | device: {device}")

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