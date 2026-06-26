"""Download and prepare datasets for the project.

Supported datasets:
  - LGG MRI Segmentation (Kaggle) — 2D brain MR slices
  - Carvana (Kaggle)              — 2D car segmentation (demo)

Usage:
  python -m src.utils.download_datasets --dataset lgg
  python -m src.utils.download_datasets --dataset carvana
  python -m src.utils.download_datasets --dataset all
"""

import argparse
import os
import sys
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MRI_DIR = PROJECT_ROOT / "MRI"
LGG_DIR = MRI_DIR / "kaggle_3m"
LGG_FILTERED = MRI_DIR / "filtered_data"
CARVANA_DIR = PROJECT_ROOT / "data" / "carvana"


def _ensure_dir(path):
    os.makedirs(str(path), exist_ok=True)
    return path


def download_lgg(force: bool = False):
    """Download LGG MRI Segmentation Dataset from Kaggle."""
    if LGG_DIR.exists() and any(LGG_DIR.iterdir()) and not force:
        print(f"LGG dataset already exists at {LGG_DIR}")
        print("  Use --force to re-download.")
        return

    print("=" * 60)
    print("Downloading LGG MRI Segmentation Dataset...")
    print("=" * 60)

    try:
        import kagglehub
        print("  Using kagglehub...")
        path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
        print(f"  Downloaded to: {path}")

        _ensure_dir(MRI_DIR)
        if LGG_DIR.exists():
            shutil.rmtree(str(LGG_DIR))
        _ensure_dir(LGG_DIR)
        for item in Path(path).iterdir():
            dst = LGG_DIR / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dst), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dst))
        print(f"  Copied to: {LGG_DIR}")

    except ImportError:
        print("  kagglehub not installed. Trying kaggle CLI...")
        ret = os.system("kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p /tmp/lgg_raw --unzip")
        if ret != 0:
            print("\n  ERROR: Could not download LGG dataset.")
            print("  Options:")
            print("    1. pip install kagglehub && python -m src.utils.download_datasets --dataset lgg")
            print("    2. Install kaggle CLI: pip install kaggle")
            print("    3. Download manually from: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation")
            sys.exit(1)
        _ensure_dir(MRI_DIR)
        if LGG_DIR.exists():
            shutil.rmtree(str(LGG_DIR))
        shutil.move("/tmp/lgg_raw", str(LGG_DIR))

    print("\n  Filtering slices with valid masks (>4% foreground)...")
    from src.utils.filter_data_mri import filter_and_copy_images
    filter_and_copy_images(str(LGG_DIR), str(LGG_FILTERED), min_foreground_ratio=0.04)

    n_cases = len([d for d in LGG_FILTERED.iterdir() if d.is_dir()]) if LGG_FILTERED.exists() else 0
    n_files = len(list(LGG_FILTERED.rglob("*.tif"))) if LGG_FILTERED.exists() else 0
    print(f"\n  Done! {n_cases} cases, {n_files} files in {LGG_FILTERED}")


def download_carvana(force: bool = False):
    """Download Carvana dataset from Kaggle."""
    if CARVANA_DIR.exists() and any(CARVANA_DIR.iterdir()) and not force:
        print(f"Carvana dataset already exists at {CARVANA_DIR}")
        return

    print("=" * 60)
    print("Downloading Carvana dataset...")
    print("=" * 60)

    try:
        import kagglehub
        path = kagglehub.dataset_download("zalando-research/carvana-image-masking")
        _ensure_dir(CARVANA_DIR.parent)
        if CARVANA_DIR.exists():
            shutil.rmtree(str(CARVANA_DIR))
        shutil.copytree(str(path), str(CARVANA_DIR))
        print(f"  Downloaded to: {CARVANA_DIR}")
    except ImportError:
        print("  Install kagglehub: pip install kagglehub")
        print("  Or download from: https://www.kaggle.com/c/carvana-image-masking-challenge")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for the project")
    parser.add_argument("--dataset", type=str, default="lgg",
                        choices=["lgg", "carvana", "all"],
                        help="Dataset to download")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if exists")
    args = parser.parse_args()

    datasets = {
        "lgg": lambda: download_lgg(args.force),
        "carvana": lambda: download_carvana(args.force),
    }

    if args.dataset == "all":
        for name, fn in datasets.items():
            print(f"\n{'#' * 60}")
            print(f"# Dataset: {name}")
            print(f"{'#' * 60}")
            fn()
    else:
        datasets[args.dataset]()

    print("\nDone!")


if __name__ == "__main__":
    main()
