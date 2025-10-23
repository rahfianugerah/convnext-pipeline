import os
import pickle
import tarfile
import numpy as np

from PIL import Image
from pathlib import Path

from sklearn.model_selection import train_test_split

# Dataset paths
ARCHIVE = Path("cifar-10-python.tar.gz")
RAW_DIR = Path("cifar-10-batches-py")
OUT_DIR = Path("data/cls")

def ensure_extracted():
    if RAW_DIR.exists():
        return
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"{ARCHIVE} not found. Please download CIFAR-10 python version first."
        )
    print(f"[INFO] Extracting {ARCHIVE} ...")
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(".")
    print("[OK] Extraction complete.")

def unpickle(file: Path):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="latin1")

def save_image(img_flat, class_name, fname, split):
    r = img_flat[0:1024].reshape(32, 32)
    g = img_flat[1024:2048].reshape(32, 32)
    b = img_flat[2048:3072].reshape(32, 32)
    img = np.dstack((r, g, b)).astype(np.uint8)
    im = Image.fromarray(img)
    out_dir = OUT_DIR / split / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    im.save(out_dir / fname)

def export_batch(batch_file: Path):
    d = unpickle(batch_file)
    return d["data"], d["labels"], d["filenames"]

def export_all():
    ensure_extracted()
    (OUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "val").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "test").mkdir(parents=True, exist_ok=True)

    meta = unpickle(RAW_DIR / "batches.meta")
    classes = meta["label_names"]
    print("[INFO] Classes:", classes)

    # Gather training data (+- ~50.000)
    X_all, y_all, fn_all = [], [], []
    for i in range(1, 6):
        X, y, fn = export_batch(RAW_DIR / f"data_batch_{i}")
        X_all.append(X)
        y_all.extend(y)
        fn_all.extend(fn)
    X_all = np.vstack(X_all)
    y_all = np.array(y_all)
    fn_all = np.array(fn_all)

    # Combine with test batch (+- ~10.000)
    dtest = unpickle(RAW_DIR / "test_batch")
    X_combined = np.vstack((X_all, dtest["data"]))
    y_combined = np.concatenate((y_all, np.array(dtest["labels"])))
    fn_combined = np.concatenate((fn_all, np.array(dtest["filenames"])))

    print("[INFO] Creating 70/15/15 split ...")
    
    # First split train vs temp (85/15)
    X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
        X_combined, y_combined, fn_combined, test_size=0.30, stratify=y_combined, random_state=42
    )
    
    # Then split temp (30%) into val/test evenly
    X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
        X_temp, y_temp, fn_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    for (arr, labels, fns, split) in [
        (X_train, y_train, fn_train, "train"),
        (X_val, y_val, fn_val, "val"),
        (X_test, y_test, fn_test, "test"),
    ]:
        print(f"[INFO] Exporting {split}/ ...")
        for img_flat, y, fname in zip(arr, labels, fns):
            save_image(img_flat, classes[y], fname, split)

    print("[OK] Done. Directory layout:")
    print(f"  {OUT_DIR}/train/<class>/*.png")
    print(f"  {OUT_DIR}/val/<class>/*.png")
    print(f"  {OUT_DIR}/test/<class>/*.png")

if __name__ == "__main__":
    export_all()