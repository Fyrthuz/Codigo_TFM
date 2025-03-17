import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) mask into a binary mask.
    
    Args:
        mask_rle (str): Run-length encoding of the mask (e.g., "2 4 10 2").
        shape (tuple): (height, width) of the output mask.
        
    Returns:
        np.ndarray: Decoded binary mask (0s and 1s).
    """
    s = mask_rle.split()
    starts, lengths = map(np.array, (s[0::2], s[1::2]))
    starts = starts.astype(int) - 1
    lengths = lengths.astype(int)
    
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape(shape)

# Load CSV
df = pd.read_csv("carvana/train_masks/train_masks.csv")

# Define image directory
image_dir = "carvana/train_images/"  # <-- Change this to your actual path

# Extract and decode masks
for idx, row in df.iterrows():
    img_name = row['img']
    rle_mask = row['rle_mask']

    # Load image using PIL to get dimensions
    img_path = os.path.join(image_dir, img_name)
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size  # Get dynamic dimensions
    except FileNotFoundError:
        print(f"Warning: Image {img_name} not found, skipping...")
        continue

    # Decode RLE mask
    mask = rle_decode(rle_mask, (height, width))

    # Save or visualize mask
    # plt.imshow(mask, cmap='gray')
    # plt.title(f"Mask for {img_name}")
    # plt.axis("off")
    # plt.show()

    # Optional: Save mask as an image using PIL
    mask_img = Image.fromarray(mask * 255)  # Convert to PIL image
    os.makedirs("carvana/train_masks/masks", exist_ok=True)
    mask_save_path = os.path.join("carvana/train_masks/masks", f"{img_name}")
    mask_img.save(mask_save_path)
