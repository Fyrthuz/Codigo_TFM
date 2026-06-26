import os
import shutil
import numpy as np
from PIL import Image


def filter_and_copy_images(source_dir, target_dir, min_foreground_ratio=0.01):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filtered_count = 0
    total_count = 0

    for case in os.listdir(source_dir):
        case_path = os.path.join(source_dir, case)
        if os.path.isdir(case_path):
            target_case_path = os.path.join(target_dir, case)
            os.makedirs(target_case_path, exist_ok=True)

            for file in os.listdir(case_path):
                if file.lower().endswith(".tif") and "_mask" not in file:
                    image_path = os.path.join(case_path, file)
                    base, ext = os.path.splitext(file)
                    mask_file = base + "_mask" + ext
                    mask_path = os.path.join(case_path, mask_file)

                    if os.path.exists(mask_path):
                        mask = Image.open(mask_path).convert("L")
                        mask_np = np.array(mask)
                        mask_bin = (mask_np > 0).astype(np.uint8)

                        total_count += 1
                        ratio = np.sum(mask_bin) / mask_bin.size
                        if ratio > min_foreground_ratio:
                            shutil.copy(image_path, os.path.join(target_case_path, file))
                            shutil.copy(mask_path, os.path.join(target_case_path, mask_file))
                            filtered_count += 1

    print(f"Total images processed: {total_count}")
    print(f"Filtered images with valid masks: {filtered_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter MRI images with valid masks")
    parser.add_argument("--source", type=str, default=None, help="Source directory")
    parser.add_argument("--target", type=str, default=None, help="Target directory")
    args = parser.parse_args()
    source = args.source or os.path.join(os.path.dirname(__file__), "../../MRI/kaggle_3m")
    target = args.target or os.path.join(os.path.dirname(__file__), "../../MRI/filtered_data")
    filter_and_copy_images(os.path.abspath(source), os.path.abspath(target))
