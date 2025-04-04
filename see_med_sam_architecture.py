import os
import glob
import yaml
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import ttach as tta
from skimage.filters import threshold_otsu
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sam = sam_model_registry["vit_b"](checkpoint="/mnt/netapp2/Store_uni/home/usc/ci/fgs/git_repo/Codigo_TFM/medsam_vit_b.pth")
sam.to(device=device)
sam.eval()  # Set model to evaluation mode

# Create the predictor
predictor = SamPredictor(sam)
print(f"MedSAM model ({'vit_b'}) loaded successfully from {'/mnt/netapp2/Store_uni/home/usc/ci/fgs/git_repo/Codigo_TFM/medsam_vit_b.pth'}.")

# Print model structure
print("MedSAM Model Structure:\n")

# Use named_modules() to see all modules, including nested ones.
for name, module in predictor.model.named_modules():
    print(f"{name} -> {module}")

#Alternative using named_children(), will only print the direct children
#for name, module in predictor.sam.named_children():
#    print(f"{name} -> {module}")