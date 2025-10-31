# export_iiit5k_dataset.py
from __future__ import annotations
from pathlib import Path
import argparse, sys, shutil
import pandas as pd

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def read_csv_any(p: Path) -> pd.DataFrame:
    # robust delimiter+encoding sniffing
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(p, encoding=enc, sep=None, engine="python")
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {p}")

def infer_cols(df: pd.DataFrame) -> tuple[str, str]:
    lower = {c.lower(): c for c in df.columns}
    img_col = next((lower[n] for n in ["image","img","filename","file","path","imgname","name"] if n in lower), None)
    lab_col = next((lower[n] for n in ["label","word","gt","text","lex","transcript"] if n in lower), None)
    if img_col and lab_col:
        return img_col, lab_col
    # fallback by content
    img_col_guess = None
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.contains(r"\.(png|jpg|jpeg)$", case=False, regex=True).mean() > 0.2:
            img_col_guess = c
            break
    if img_col_guess is None:
        img_col_guess = df.columns[0]
    lab_col_guess = next((c for c in df.columns if c != img_col_guess), df.columns[1])
    return img_col_guess, lab_col_guess

def find_csv_pair(raw_root: Path) -> tuple[Path, Path]:
    # Look anywhere under raw_root for the known file names
    all_csvs = list(raw_root.rglob("*.csv"))
    if not all_csvs:
        raise SystemExit(f"[ERR] No CSV files under: {raw_root}")

    # Prefer exact file names if present
    by_name = {c.name.lower(): c for c in all_csvs}
    exact = [("traindata.csv","testdata.csv"),
             ("trainCharBound.csv".lower(),"testCharBound.csv".lower())]
    for a, b in exact:
        if a in by_name and b in by_name:
            return by_name[a], by_name[b]

    # Otherwise, heuristically pick the best “train” and “test” CSVs
    def score(p: Path) -> int:
        s = p.name.lower()
        sc = 0
        if "train" in s: sc += 2
        if "test"  in s: sc += 2
        if "data"  in s: sc += 1
        if "bound" in s: sc += 1
        return sc
    trains = sorted([p for p in all_csvs if "train" in p.name.lower()], key=score, reverse=True)
    tests  = sorted([p for p in all_csvs if "test"  in p.name.lower()], key=score, reverse=True)
    if trains and tests:
        return trains[0], tests[0]

    # Last resort: tell the user what we saw
    sample = "\n".join(" - " + str(p.relative_to(raw_root)) for p in all_csvs[:10])
    raise SystemExit("[ERR] Could not pick train/test CSVs automatically.\nCSV sample found:\n" + sample)

def index_images(raw_root: Path) -> dict[str, Path]:
    # Map basename -> absolute path, recursively
    idx: dict[str, Path] = {}
    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx[p.name] = p
    if not idx:
        raise SystemExit(f"[ERR] No images found anywhere under: {raw_root}")
    return idx

def resolve_img(idx: dict[str, Path], name: str) -> Path | None:
    base = Path(name).name
    if base in idx:
        return idx[base]
    stem = Path(base).stem
    for ext in IMG_EXTS:  # handle .jpg vs .png mismatches
        alt = stem + ext
        if alt in idx:
            return idx[alt]
    return None

def write_split(out_root: Path, split: str, df: pd.DataFrame, img_col: str, lab_col: str, img_index: dict[str, Path]) -> tuple[int,int,int]:
    imgdir = out_root / split / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    lines = []
    copied = 0
    missing = 0
    for img, lab in zip(df[img_col].astype(str), df[lab_col].astype(str)):
        src = resolve_img(img_index, img)
        if src is None:
            missing += 1
            continue
        dst = imgdir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
        # labels.txt uses just the filename (no folder)
        lines.append(f"{src.name}\t{lab}")
    (out_root / split / "labels.txt").write_text("\n".join(lines), encoding="utf-8")
    return len(lines), copied, missing

def main():
    ap = argparse.ArgumentParser("Export IIIT5K into train/test (no val) from data_root/raw/**")
    ap.add_argument("--data_root", default="./data/iiit5k_ocr", help="Root that contains 'raw/' and where train/test will be written.")
    args = ap.parse_args()

    DATA = Path(args.data_root).resolve()
    RAW  = DATA / "raw"
    if not RAW.exists():
        raise SystemExit(f"[ERR] Missing folder: {RAW}\nPlace the unzipped dataset under this 'raw' folder.")

    print("[INFO] Scanning CSVs under:", RAW)
    csv_train, csv_test = find_csv_pair(RAW)
    print("[INFO] Using CSVs:")
    print("  train:", csv_train.relative_to(RAW))
    print("  test :", csv_test.relative_to(RAW))

    df_tr = read_csv_any(csv_train)
    df_te = read_csv_any(csv_test)
    ti, tl = infer_cols(df_tr)
    vi, vl = infer_cols(df_te)
    print(f"[INFO] Columns -> train({ti}, {tl})  test({vi}, {vl})")

    print("[INFO] Indexing images recursively under:", RAW)
    img_idx = index_images(RAW)
    print(f"[INFO] Found {len(img_idx)} images")

    ntr, cpy_tr, miss_tr = write_split(DATA, "train", df_tr, ti, tl, img_idx)
    nte, cpy_te, miss_te = write_split(DATA, "test",  df_te, vi, vl, img_idx)
    if ntr + nte == 0:
        raise SystemExit("[ERR] Wrote 0 labels — CSV filenames may not match any images under 'raw'.")

    # Build charset from both splits
    texts = []
    for sp in ("train","test"):
        f = DATA/sp/"labels.txt"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    texts.append(line.split("\t",1)[1])
    charset = "".join(sorted(set("".join(texts))))
    (DATA/"charset.txt").write_text(charset, encoding="utf-8")

    print(f"[OK] train: labels={ntr}, copied={cpy_tr}, missing={miss_tr}")
    print(f"[OK] test : labels={nte}, copied={cpy_te}, missing={miss_te}")
    print(f"[OK] charset size:", len(charset))
    print("[DONE] Exported to:", DATA)

if __name__ == "__main__":
    main()