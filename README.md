## ConvNeXt + Optuna Multi Task Pipeline
**Unified Training Framework for Image Classification, Object Detection, and OCR Recognition**

### Overview
<p align="justify">
This repository implements a <b>multi-task AutoML pipeline</b> using <b>Optuna</b> for hyperparameter optimization, powered by ConvNeXt backbones. It supports three modes:
</p>

1. **Classification** – ConvNeXt fine-tuning with standard metrics like Accuracy, Precision, Recall, F1-Score and Specificity (If binary classification, 2 labels only)
2. **Object Detection** – Faster-RCNN with ConvNeXt backbone, evaluated via mAP@0.5:0.95 and mean IoU  
3. **OCR (Optical Character Recognition)** – ConvNeXt + BiLSTM + CTC head for sequence text recognition, evaluated via CER/WER  

Each task runs a **complete train/val/test pipeline** with **Optuna-based hyperparameter search**, automatic checkpointing, and test-set evaluation.

### Environment Setup

Using Conda (recommended)
```bash
conda create -n convnext python=3.10 -y
conda activate convnext
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y -c conda-forge optuna opencv
pip install timm scikit-learn torchmetrics jiwer tqdm
```

### Dataset Layouts
**Image Classification**
```
data_root/
 ├── train/<class>/*.jpg
 ├── val/<class>/*.jpg
 └── test/<class>/*.jpg
```
**Object Detection**
```
data_root/
 ├── images/*.jpg
 ├── train.csv
 ├── val.csv
 └── test.csv

Each csv must contain:
filename,xmin,ymin,xmax,ymax,label
```
**OCR (Word Recognition)**
```
data_root/
 ├── train/
 │    ├── images/*.png
 │    └── labels.txt     # e.g. img001.png<TAB>HELLO
 ├── val/
 │    ├── images/*.png
 │    └── labels.txt
 └── test/
      ├── images/*.png
      └── labels.txt
```
You can generate `labels.txt` from `train_gt.txt–style` datasets using:
```bash
python rec_to_labels_txt.py
```

### Run Model Training
General Syntax
```bash
python main.py --data_root <DATA_PATH> --theme <classification|object|ocr> [options]
```
Classification Example
```bash
python main.py \
  --theme classification \
  --data_root ./data/cls \
  --epochs 10 \
  --n_trials 5 \
  --out_dir ./artifacts/cls_run
```
Object Detection Example
```bash
python main.py \
  --theme object \
  --data_root ./data/det \
  --epochs 15 \
  --n_trials 3 \
  --det_size 640 \
  --out_dir ./artifacts/det_run
```
OCR Example
```bash
python main.py \
  --theme ocr \
  --data_root ./data/ocr \
  --epochs 15 \
  --n_trials 5 \
  --ocr_h 32 --ocr_w 256 \
  --out_dir ./artifacts/ocr_run
```

---

### Arguments Explained

```bash
--data_root
```
Path to your dataset directory. This is required for all modes (`classification`, `object`, and `ocr`).  
Example: `--data_root ./data/cls`

```bash
--out_dir
```
Directory where all models, logs, and results will be saved.  
Default: `./artifacts`

```bash
--theme
```
Specifies which task to run.  
Options: `classification`, `object`, or `ocr`  
Default: `classification`

```bash
--epochs
```
Number of training epochs per Optuna trial.  
Default: `3`

**--n_trials**  
Number of Optuna hyperparameter trials to run. Each trial tests a different parameter configuration.  
Default: `1`

**--seed**  
Random seed for full reproducibility.  
Default: `42`

```bash
--workers
```
Number of DataLoader workers for parallel data loading. Set `0` for Windows; higher values (e.g., 4 or 8) for Linux to speed up loading.  
Default: `0`

```bash
--img_size
```
Image resize dimension for the classification pipeline.  
Default: `224`

```bash
--det_size
```
Image resize dimension for the object detection pipeline.  
Default: `640`

```bash
--ocr_h

or/ [dont add this]

--ocr_w
```
Image height and width for the OCR pipeline. These define the resized input size of each text image before training.  
Defaults: `--ocr_h 32`, `--ocr_w 256`

```bash
--charset
```
Character set used by the OCR label encoder. You can customize this if your dataset includes special symbols.  
Default: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`

```bash
--study_name
```
Optional name for the Optuna study. Useful if you want to resume or track experiments in a database.  
Default: `None`

```bash
--storage
```
Optuna storage path for saving study results persistently.  
Example: `--storage sqlite:///study.db`  
Default: `None`

---

### Notes
- All tasks share the same pipeline logic and Optuna hyperparameter search.
- Default ConvNeXt variant is **`convnext_tiny`** (change manually inside the code).
- The pipeline **automatically runs test-set evaluation** after the best trial finishes.
- Results are saved under:  
  `artifacts/<theme>/best_model.pth`, `best_params.json`, and `test_metrics.json`.

### Test-Set Evaluation
Once training finishes, the pipeline automatically evaluates on the test split using the best checkpoint (`best_model.pth`).
Results are printed and written to `test_metrics.json`.

To re-run test evaluation manually:
```bash
python main.py --data_root ./data/cls --theme classification --out_dir ./artifacts
```

### Outputs & Artifacts
```bash
artifacts/<theme>/
 ├── best_model.pth
 ├── best_params.json
 ├── study.db              # Optuna study (if storage enabled)
 ├── test_metrics.json
 └── logs / prints
```




