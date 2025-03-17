import os
import shutil
import numpy as np
from PIL import Image

def filter_and_copy_images(source_dir, target_dir):
    """
    Filters images based on masks and copies only those with at least one foreground pixel (value=1).
    
    Args:
        source_dir (str): Path to the root directory containing cases with images and masks.
        target_dir (str): Path to the output directory where filtered images/masks will be saved.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filtered_count = 0
    total_count = 0

    # Iterate over case folders
    for case in os.listdir(source_dir):
        case_path = os.path.join(source_dir, case)
        if os.path.isdir(case_path):
            target_case_path = os.path.join(target_dir, case)
            os.makedirs(target_case_path, exist_ok=True)

            # Process each file in the case folder
            for file in os.listdir(case_path):
                # Look for .tif files that are not masks (i.e., do not contain '_mask')
                if file.lower().endswith('.tif') and '_mask' not in file:
                    image_path = os.path.join(case_path, file)
                    base, ext = os.path.splitext(file)
                    mask_file = base + '_mask' + ext
                    mask_path = os.path.join(case_path, mask_file)

                    if os.path.exists(mask_path):
                        # Load mask and check for nonzero pixels
                        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
                        mask_np = np.array(mask)
                        mask_bin = (mask_np > 0).astype(np.uint8)  # Binarize mask

                        total_count += 1
                        if np.sum(mask_bin)/(mask_bin.shape[0]*mask_bin.shape[1]) > 0.07:  # If mask contains any 1s
                            # Copy image and mask to target directory
                            shutil.copy(image_path, os.path.join(target_case_path, file))
                            shutil.copy(mask_path, os.path.join(target_case_path, mask_file))
                            filtered_count += 1
                        else:
                            print(f"Skipping {file} - mask is empty.")

    print(f"Total images processed: {total_count}")
    print(f"Filtered images with valid masks: {filtered_count}")

# Example usage:
source_directory = "/mnt/netapp2/Store_uni/home/usc/ci/fgs/Codigo/MRI/kaggle_3m"
target_directory = "/mnt/netapp2/Store_uni/home/usc/ci/fgs/Codigo/MRI/filtered_data"

filter_and_copy_images(source_directory, target_directory)
