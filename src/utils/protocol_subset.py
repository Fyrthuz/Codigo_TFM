"""Build the 4%-foreground evaluation protocol (thesis reproduction).

The thesis reported its analyses on a 372-image subset: the slices with
more than 4% tumour foreground in the raw LGG dataset (3,929 slices ->
1,373 with tumour -> 372 above 4%). The main pipeline results use the
1%-filtered dataset (1,060 images), so this utility rebuilds the
thesis protocol for comparison:

  - selects the slices above the threshold,
  - links them (hardlinks, no extra disk usage) into a new dataset root,
  - writes the index files consumed by the pipelines:
      protocol_all.json      -> every selected slice; used to evaluate the
                                UNet over the full 372-image set,
      protocol_val_test.json -> slices of val+test patients, in the same
                                patient split as the main protocol
                                (seed=42); used to evaluate UniVerSeg,
                                which needs a context and is therefore
                                evaluated on unseen patients only.

Example:
    python -m src.utils.protocol_subset --threshold 0.04
    python -m src.pipelines.run_unet --config configs/pipeline_2d_4pct.yaml \\
        --test-indices MRI/filtered_data_4pct/protocol_all.json
    python -m src.pipelines.run_foundation --config configs/foundation_universeg_4pct.yaml \\
        --test-indices MRI/filtered_data_4pct/protocol_val_test.json --context-size 64
"""

import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np
from PIL import Image

from src.utils.dataset import load_test_indices, recover_image_mask_pairs, split_by_patient


def build_subset(src_root: str, out_dir: str, threshold: float, force: bool = False) -> list:
    """Select slices above `threshold` foreground and hardlink them into out_dir.

    Returns the list of selected indices into the source pair list.
    """
    pairs, _ = recover_image_mask_pairs(src_root)
    selected = []
    for i, (_ip, mp) in enumerate(pairs):
        mask = np.asarray(Image.open(mp), dtype=np.float32) / 255
        if (mask > 0.5).mean() > threshold:
            selected.append(i)

    has_files = os.path.isdir(out_dir) and any(
        f.lower().endswith(".tif") for _root, _dirs, files in os.walk(out_dir) for f in files
    )
    if has_files and not force:
        print(f"Reusing existing subset at {out_dir} (use --force to rebuild)")
    else:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        for i in selected:
            image_path, mask_path = pairs[i]
            case = os.path.basename(os.path.dirname(image_path))
            os.makedirs(os.path.join(out_dir, case), exist_ok=True)
            for src in (image_path, mask_path):
                dst = os.path.join(out_dir, case, os.path.basename(src))
                if not os.path.exists(dst):
                    os.link(src, dst)
        print(f"Linked {2 * len(selected)} files into {out_dir}")

    return selected


def build_indices(src_root: str, out_dir: str, test_indices_path: str, seed: int = 42) -> dict:
    """Write protocol_all.json / protocol_val_test.json inside out_dir."""
    pairs_src, pids_src = recover_image_mask_pairs(src_root)
    test_idx = load_test_indices(test_indices_path)

    train, val, test = split_by_patient(pids_src, seed=seed)
    if sorted(test) != sorted(test_idx):
        raise SystemExit(
            "Patient split mismatch: split_by_patient(seed=%d) does not reproduce %s. "
            "The dataset iteration order changed." % (seed, test_indices_path)
        )

    group = {}
    for i in train:
        group[pids_src[i]] = "train"
    for i in val:
        group[pids_src[i]] = "val"
    for i in test:
        group[pids_src[i]] = "test"

    group_by_file = {
        (os.path.basename(os.path.dirname(ip)), os.path.basename(ip)): group[pids_src[i]]
        for i, (ip, _mp) in enumerate(pairs_src)
    }

    pairs_out, _ = recover_image_mask_pairs(out_dir)
    groups_out = []
    for ip, _mp in pairs_out:
        key = (os.path.basename(os.path.dirname(ip)), os.path.basename(ip))
        if key not in group_by_file:
            raise SystemExit(f"Slice {key} not found in the source dataset; rebuild with --force.")
        groups_out.append(group_by_file[key])

    all_idx = list(range(len(pairs_out)))
    val_test_idx = [k for k, g in enumerate(groups_out) if g in ("val", "test")]

    with open(os.path.join(out_dir, "protocol_all.json"), "w") as f:
        json.dump(all_idx, f)
    with open(os.path.join(out_dir, "protocol_val_test.json"), "w") as f:
        json.dump(val_test_idx, f)

    counts = Counter(groups_out)
    print(f"Slices: {len(pairs_out)} total | train {counts['train']} | val {counts['val']} | test {counts['test']}")
    print(f"Indices written: protocol_all.json ({len(all_idx)}), protocol_val_test.json ({len(val_test_idx)})")
    return {"all": all_idx, "val_test": val_test_idx}


def main():
    parser = argparse.ArgumentParser(description="Build the thesis 4% foreground protocol (372 images)")
    parser.add_argument("--threshold", type=float, default=0.04, help="Foreground ratio threshold")
    parser.add_argument("--src-root", default="./MRI/filtered_data",
                        help="Source dataset (1%%-filtered by default)")
    parser.add_argument("--out-dir", default="./MRI/filtered_data_4pct",
                        help="Output subset root (hardlinked)")
    parser.add_argument("--test-indices", default="./test_indices.json",
                        help="Main-protocol split file used to assign val/test patients")
    parser.add_argument("--seed", type=int, default=42, help="Patient split seed")
    parser.add_argument("--force", action="store_true", help="Rebuild the subset even if it exists")
    args = parser.parse_args()

    build_subset(args.src_root, args.out_dir, args.threshold, force=args.force)
    build_indices(args.src_root, args.out_dir, args.test_indices, seed=args.seed)


if __name__ == "__main__":
    main()
