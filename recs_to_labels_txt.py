# Save as: rec_to_labels_txt.py
# Example for datasets whose gt lines look like: "path/to/img.png\tTHE_TEXT"
import shutil

from pathlib import Path

SRC_LIST = Path("path/to/train_gt.txt") # Change per dataset
SRC_IMG_ROOT = Path("path/to/images") # Base image folder if paths are relative
OUT = Path("data/ocr/train")
(OUT/"images").mkdir(parents=True, exist_ok=True)

def copy_and_strip(line):
    line = line.rstrip("\n")
    if not line: return None
    
    # Handle separators: \t or ' ' or ',' — adapt if needed
    if "\t" in line: fp, txt = line.split("\t",1)
    elif "," in line: fp, txt = line.split(",",1)
    else: fp, txt = line.split(" ",1)
    
    src = (SRC_IMG_ROOT/fp) if not Path(fp).exists() else Path(fp)
    
    dst = OUT/"images"/Path(fp).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    
    return f"{dst.name}\t{txt}"

lines = [copy_and_strip(l) for l in SRC_LIST.read_text(encoding="utf-8").splitlines()]
lines = [l for l in lines if l]

(OUT/"labels.txt").write_text("\n".join(lines), encoding="utf-8")
print("Wrote", OUT/"labels.txt")