from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parent.parent

# Data paths
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROF_DIR      = PROCESSED_DIR / "prof_components_extracted"
ACC_DIR       = PROCESSED_DIR / "acc_components_extracted"
MAPPINGS_DIR  = ROOT / "data" / "mappings"

# Output
OUTPUT_DIR  = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "plots"

if __name__ != "__main__":
    print(f"config.py loaded from: {_THIS_FILE}")
    print(f"ROOT: {ROOT}")