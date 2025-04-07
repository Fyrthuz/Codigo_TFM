import os
import glob
import yaml
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import ttach as tta
from skimage.filters import threshold_otsu
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --- Required for UniVerSeg ---
# Installation: pip install git+https://github.com/JJGO/UniverSeg.git
try:
    from universeg import universeg
    print("UniVerSeg imported but instantiated")
except ImportError:
    print("Error: 'universeg' package not found.")
    print("Please install it: pip install git+https://github.com/JJGO/UniverSeg.git")
    exit()
# --------------------------

# Import custom modules (Keep if MCDropout/NoisyInference are still desired)
from mc_dropout import MCDropout
from noise_inference import NoisyInference

# pydensecrf imports
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

print("All imports correctly done")

# ----------------- Helper functions -----------------
# [recover_image_mask_pairs, save_image, compute_iou, compute_dice, compute_metrics remain unchanged]
# ... (Keep the original helper functions here: recover_image_mask_pairs, save_image, compute_iou, compute_dice, compute_metrics) ...
def recover_image_mask_pairs(root_dir):
    # [Unchanged from original code]
    image_mask_pairs = []
    for case in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case)
        if os.path.isdir(case_path):
            for file in os.listdir(case_path):
                if file.lower().endswith('.tif') and '_mask' not in file:
                    image_path = os.path.join(case_path, file)
                    base, ext = os.path.splitext(file)
                    mask_file = base + '_mask' + ext
                    mask_path = os.path.join(case_path, mask_file)
                    if os.path.exists(mask_path):
                        image_mask_pairs.append((image_path, mask_path))
                    else:
                        print(f'Warning: Mask not found for image {image_path}')
    return image_mask_pairs


import PIL.Image
import numpy as np
import torch

def save_image(array, path):
    """Saves a numpy array or tensor as an image, correctly handling masks."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.squeeze(array) # Remove single dimensions like batch or channel

    # --- Scaling Logic ---
    # 1. Handle boolean input: Convert True/False -> 255/0
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255

    # 2. Handle float input (probabilities/uncertainty): Scale to 0-255
    elif np.issubdtype(array.dtype, np.floating):
        if array.min() >= 0 and array.max() <= 1: # Assume probability map [0, 1]
            array = (array * 255).astype(np.uint8)
        else: # Handle other float ranges (like uncertainty, logits) - normalize & scale
            # Basic normalization (might clip outliers)
            array_min, array_max = np.percentile(array, [2, 98])
            array = np.clip(array, array_min, array_max)
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)
            array = (array * 255).astype(np.uint8)

    # 3. Handle integer input (like uint8 masks with 0/1):
    elif np.issubdtype(array.dtype, np.integer):
        unique_vals = np.unique(array)
        # Check if it looks like a binary 0/1 mask
        if np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [1]) or np.array_equal(unique_vals, [0, 1]):
            array = array.astype(np.uint8) * 255
        # Else: Assume it's already scaled (e.g., 0-255) or a label map - save as is
        else:
            array = array.astype(np.uint8) # Ensure uint8

    # --- Saving Logic ---
    # Now, array should be uint8 and scaled appropriately for visualization
    if array.ndim == 2:
        image = PIL.Image.fromarray(array, mode='L') # Save as grayscale
        image.save(path)
        return
    elif array.ndim == 3:
        # Try to determine if it's HWC (Height, Width, Channels)
        if array.shape[-1] in [1, 3]: # Likely HWC
             if array.shape[-1] == 1: # Grayscale HWC
                  image = PIL.Image.fromarray(array[:, :, 0], mode='L')
             else: # RGB HWC
                  image = PIL.Image.fromarray(array, mode='RGB')
             image.save(path)
             return
        else: # Try assuming CHW if HWC failed (less common for saving)
             if array.shape[0] in [1, 3]:
                 array = np.transpose(array, (1, 2, 0)) # Convert CHW to HWC
                 # Retry saving logic from HWC above
                 if array.shape[-1] == 1: # Grayscale HWC
                     image = PIL.Image.fromarray(array[:, :, 0], mode='L')
                 else: # RGB HWC
                     image = PIL.Image.fromarray(array, mode='RGB')
                 image.save(path)
                 return

    raise ValueError(f"Unsupported array shape for saving after processing: {array.shape}, dtype: {array.dtype}")


def compute_iou(pred, target):
    # [Unchanged from original code]
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    # Ensure boolean or 0/1 input
    pred = (pred > 0.5).astype(bool) if pred.dtype != bool else pred
    target = (target > 0.5).astype(bool) if target.dtype != bool else target

    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    intersection_sum = intersection.sum()
    union_sum = union.sum()
    if union_sum == 0:
        return 1.0 if intersection_sum == 0 else 0.0 # Both empty
    return intersection_sum / union_sum

def compute_dice(pred, target):
    # [Unchanged from original code]
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    # Ensure boolean or 0/1 input
    pred = (pred > 0.5).astype(bool) if pred.dtype != bool else pred
    target = (target > 0.5).astype(bool) if target.dtype != bool else target

    intersection = np.logical_and(pred, target)
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum + target_sum == 0:
        return 1.0 # Both empty
    return (2 * intersection.sum()) / (pred_sum + target_sum + 1e-8) # Epsilon for safety


def compute_metrics(prob, gt_mask, epsilon=1e-8):
    # Ensure prob is numpy and squeezed
    if isinstance(prob, torch.Tensor):
        prob = prob.detach().cpu().numpy()
    prob = np.squeeze(prob)

    # Ensure gt_mask is numpy, squeezed, and boolean/int
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    gt_mask = np.squeeze(gt_mask)
    gt_mask = (gt_mask > 0.5).astype(np.int64) # Ensure binary 0/1

    # Ensure prob is clipped correctly
    prob = np.clip(prob, epsilon, 1 - epsilon)

    prob_flat = prob.flatten()
    gt_flat = gt_mask.flatten()

    if len(prob_flat) != len(gt_flat):
        raise ValueError(f"Shape mismatch: prob_flat {prob_flat.shape}, gt_flat {gt_flat.shape}")

    # Negative Log Likelihood
    nll = -np.mean(gt_flat * np.log(prob_flat) + (1 - gt_flat) * np.log(1 - prob_flat))

    # Brier Score
    brier = np.mean((prob_flat - gt_flat) ** 2)

    # Predicted mask
    pred_mask_flat = (prob_flat > 0.5).astype(np.int64)

    # Confusion matrix components
    tp = np.sum((pred_mask_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_mask_flat == 1) & (gt_flat == 0))
    tn = np.sum((pred_mask_flat == 0) & (gt_flat == 0))
    fn = np.sum((pred_mask_flat == 0) & (gt_flat == 1))

    # Accuracy, Precision, Recall
    total = len(gt_flat)
    accuracy = (tp + tn) / (total + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon) # Also known as Sensitivity

    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, 11)
    try:
        bin_indices = np.digitize(prob_flat, bin_edges[1:-1], right=True) # Indices 0 to 10
    except ValueError as e:
         print(f"Error in np.digitize: {e}")
         print(f"prob_flat sample: {prob_flat[:10]}")
         print(f"bin_edges: {bin_edges}")
         ece = np.nan # Cannot compute ECE
    else:
        ece = 0.0
        for i in range(10): # Bins 0 to 9, corresponding to intervals
            mask = (bin_indices == i)
            bin_size = np.sum(mask)
            if bin_size == 0:
                continue
            conf = np.mean(prob_flat[mask])
            acc = np.mean((pred_mask_flat[mask] == gt_flat[mask]).astype(float))
            ece += np.abs(acc - conf) * bin_size
        ece /= (len(prob_flat) + epsilon)

    return {
        'nll': nll,
        'ece': ece,
        'brier': brier,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


import torch
import numpy as np

def universeg_inference(model: universeg, image_tensor: torch.Tensor, 
                      support_images: torch.Tensor, support_masks: torch.Tensor):
    # Get model prediction
    prediction = model(
        image_tensor,  # (B, 1, H, W)
        support_images,  # (B, S, 1, H, W)
        support_masks  # (B, S, 1, H, W)
    )  # -> (B, 1, H, W)
    
    # Convert to probabilities using sigmoid
    prob = torch.sigmoid(prediction)
    # Create binary mask by thresholding at 0.5
    mask = (prob > 0.5).float()

    mask = mask.squeeze().squeeze().cpu().numpy()

    prob = prob.squeeze().squeeze().detach().cpu().numpy()
    
    return prob, mask

# --- Normal Inference (using UniVerSeg) ---
def normal_inference(model: universeg, image_tensor: torch.Tensor, 
                    support_images: torch.Tensor, support_masks: torch.Tensor):
    # Simply call universeg_inference which now returns both prob and mask
    prob, mask = universeg_inference(model, image_tensor, support_images, support_masks)

    return prob, mask


# [dynamic_threshold_multiclass, weighted_average_with_uncertainty, refine_with_crf_uncertainty remain unchanged]
# ... (Keep these functions as they operate on probability/uncertainty maps) ...
def dynamic_threshold_multiclass(prob_maps, method="otsu", percentile=50, k=0.5, epsilon=1e-6):
    # [Unchanged from original code]
    prob_maps = np.squeeze(prob_maps) # Make sure it handles (1, H, W) inputs
    if prob_maps.ndim == 2:
        try:
             if method.lower() == "otsu":
                  if np.all(prob_maps == prob_maps.flat[0]): # Handle flat images for Otsu
                       threshold = 0.5 # Or np.mean(prob_maps)
                  else:
                       threshold = threshold_otsu(prob_maps)
             elif method.lower() == "percentile":
                  threshold = np.percentile(prob_maps, percentile)
             elif method.lower() == "mean_std":
                  mean_val = np.mean(prob_maps)
                  std_val = np.std(prob_maps)
                  threshold = mean_val + k * std_val
             else:
                  raise ValueError("Unknown method. Choose from 'otsu', 'percentile', or 'mean_std'.")
        except Exception as e:
             print(f"Warning: Dynamic thresholding failed ({e}). Falling back to 0.5")
             threshold = 0.5
        return np.array([threshold])

    elif prob_maps.ndim == 3: # Multiclass - Less applicable for current binary setup
        num_classes = prob_maps.shape[0]
        thresholds = np.zeros(num_classes)
        for c in range(num_classes):
            class_probs = prob_maps[c]
            try:
                 if method.lower() == "otsu":
                      if np.all(class_probs == class_probs.flat[0]):
                           thresholds[c] = 0.5
                      else:
                           thresholds[c] = threshold_otsu(class_probs)
                 elif method.lower() == "percentile":
                      thresholds[c] = np.percentile(class_probs, percentile)
                 elif method.lower() == "mean_std":
                      mean_val = np.mean(class_probs)
                      std_val = np.std(class_probs)
                      thresholds[c] = mean_val + k * std_val
                 else:
                      raise ValueError("Unknown method. Choose from 'otsu', 'percentile', or 'mean_std'.")
            except Exception as e:
                 print(f"Warning: Dynamic thresholding for class {c} failed ({e}). Falling back to 0.5")
                 thresholds[c] = 0.5
        return thresholds
    else:
        raise ValueError(f"prob_maps must be either a 2D (binary) or 3D (multiclass) array. Got shape {prob_maps.shape}")

def weighted_average_with_uncertainty(mc_mean, mc_uncert, tta_mean, tta_uncert, noise_mean, noise_uncert,
                                     weighting_method="inverse", beta=1.0, alpha=1.0,
                                     threshold_method="otsu", percentile=50, k=0.5, epsilon=1.e-6):
    # [Unchanged logic, but ensure inputs are numpy arrays]
    mc_mean = np.squeeze(np.asarray(mc_mean))
    tta_mean = np.squeeze(np.asarray(tta_mean))
    noise_mean = np.squeeze(np.asarray(noise_mean))
    mc_uncert = np.squeeze(np.asarray(mc_uncert))
    tta_uncert = np.squeeze(np.asarray(tta_uncert))
    noise_uncert = np.squeeze(np.asarray(noise_uncert))

    prob_maps = np.stack([mc_mean, tta_mean, noise_mean], axis=0)
    uncertainty_maps = np.stack([mc_uncert, tta_uncert, noise_uncert], axis=0)

    # Normalize uncertainty maps for some weighting methods if needed (optional)
    # uncertainty_maps = (uncertainty_maps - np.min(uncertainty_maps, axis=(1,2), keepdims=True)) / \
    #                    (np.ptp(uncertainty_maps, axis=(1,2), keepdims=True) + epsilon)

    if weighting_method.lower() == "inverse":
        # Add small epsilon to avoid division by zero, especially if uncertainty can be 0
        weights = 1.0 / (uncertainty_maps + epsilon)
    elif weighting_method.lower() == "exponential":
        weights = np.exp(-beta * uncertainty_maps)
    elif weighting_method.lower() == "powerlaw":
        # Ensure uncertainty is scaled [0, 1] if using powerlaw
        scaled_uncert = (uncertainty_maps - np.min(uncertainty_maps)) / (np.ptp(uncertainty_maps) + epsilon)
        weights = (1.0 - scaled_uncert) ** alpha
    else:
        raise ValueError("Unsupported weighting method.")

    # Normalize weights across methods for each pixel
    weights_sum = np.sum(weights, axis=0, keepdims=True)
    weights = weights / (weights_sum + epsilon) # Add epsilon for stability

    # Compute weighted average probability and uncertainty
    consensus_prob = np.sum(prob_maps * weights, axis=0)
    # Weighted average of uncertainty - interpretation might vary
    consensus_uncertainty = np.sum(uncertainty_maps * weights, axis=0)
    # Alternative: Variance of weighted predictions (more complex)
    # consensus_uncertainty = np.sum(weights * (prob_maps - consensus_prob[np.newaxis,...])**2, axis=0)

    # Thresholding the consensus probability map
    if consensus_prob.ndim == 2:
        num_classes = 1
        H, W = consensus_prob.shape
    elif consensus_prob.ndim == 3 and consensus_prob.shape[0]==1: # Handle (1, H, W)
         consensus_prob = consensus_prob.squeeze(0)
         num_classes = 1
         H, W = consensus_prob.shape
    elif consensus_prob.ndim == 3: # Multiclass
        num_classes = consensus_prob.shape[0]
        H, W = consensus_prob.shape[1:]
    else:
         raise ValueError(f"Unsupported consensus_prob shape: {consensus_prob.shape}")

    if threshold_method == 'naive':
        if num_classes == 1:
            consensus_mask = (consensus_prob > 0.5).astype(np.uint8)
        else: # Multiclass
            consensus_mask = np.argmax(consensus_prob, axis=0).astype(np.uint8)
    else: # Dynamic thresholding
        thresholds = dynamic_threshold_multiclass(consensus_prob, method=threshold_method, percentile=percentile, k=k)

        if num_classes == 1:
             # thresholds will be array of size 1
            consensus_mask = (consensus_prob > thresholds[0]).astype(np.uint8)
        else: # Multiclass
             # Apply threshold per class, then take argmax
             class_masks = np.zeros_like(consensus_prob, dtype=np.uint8)
             for c in range(num_classes):
                  class_masks[c] = (consensus_prob[c] > thresholds[c]).astype(np.uint8)
             # Resolve overlaps by taking the class with the highest probability *above* its threshold
             # Or simply take the argmax of the thresholded masks (might lead to empty areas if no class passes threshold)
             # A safer approach: Apply threshold, then take argmax of original probabilities where *any* class passed threshold
             valid_pixels = np.any(class_masks, axis=0)
             consensus_mask = np.zeros((H,W), dtype=np.uint8)
             probs_above_thresh = consensus_prob * class_masks # Zero out probs below threshold
             consensus_mask[valid_pixels] = np.argmax(probs_above_thresh[:, valid_pixels], axis=0).astype(np.uint8)
             # Note: Needs careful check for multiclass logic based on desired behavior

    return consensus_prob, consensus_uncertainty, consensus_mask

def refine_with_crf_uncertainty(image, prob_map, uncertainty_map,
                                sdims=(5, 5), schan=(5, 5, 5),
                                n_iters=5, epsilon=1e-8):
    # --- Handle image input ---
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        image = np.squeeze(image)
        if image.ndim == 2:  # grayscale
            image = np.stack([image]*3, axis=-1)
        elif image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))  # CxHxW -> HxWxC
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)

    image = np.ascontiguousarray(image.astype(np.uint8))  # Ensure uint8 and contiguous

    # --- Process prob_map and uncertainty_map ---
    prob_map = np.asarray(np.squeeze(prob_map))
    uncertainty_map = np.asarray(np.squeeze(uncertainty_map))

    if prob_map.ndim == 2:  # Binary case
        prob_stack = np.stack([1 - prob_map, prob_map], axis=0)
        n_classes = 2
    elif prob_map.ndim == 3:
        n_classes = prob_map.shape[0]
        prob_stack = prob_map
    else:
        raise ValueError(f"prob_map must be 2D or 3D, got shape {prob_map.shape}")

    H, W = prob_stack.shape[1:]

    # --- Sanity checks ---
    if uncertainty_map.shape != (H, W):
        raise ValueError(f"Shape mismatch: prob_map ({H},{W}) vs uncertainty_map {uncertainty_map.shape}")
    if image.shape[:2] != (H, W):
        raise ValueError(f"Shape mismatch: prob_map ({H},{W}) vs image {image.shape[:2]}")

    # --- CRF Setup ---
    d = dcrf.DenseCRF2D(W, H, n_classes)

    prob_stack_clamped = np.clip(prob_stack, epsilon, 1 - epsilon)
    unary = unary_from_softmax(prob_stack_clamped).astype(np.float32)

    # Normalize and flatten uncertainty
    norm_uncert = (uncertainty_map - np.min(uncertainty_map)) / (np.ptp(uncertainty_map) + epsilon)
    norm_uncert_flat = norm_uncert.flatten().astype(np.float32)

    # Uniform unary (for high uncertainty)
    uniform_unary = np.ones((n_classes, H * W), dtype=np.float32) / n_classes
    uniform_unary = -np.log(uniform_unary + epsilon)

    # Blend unary with uniform unary
    uncertainty_factor = norm_uncert_flat[np.newaxis, :]
    adjusted_unary = (1 - uncertainty_factor) * unary + uncertainty_factor * uniform_unary

    d.setUnaryEnergy(adjusted_unary)

    # --- Pairwise potentials ---
    if image.ndim != 3 or image.shape[2] != len(schan):
        print(f"Warning: Adjusting schan from {schan} to match image with {image.shape[2]} channels.")
        schan = tuple([schan[0]] * image.shape[2])

    pairwise_bilateral = create_pairwise_bilateral(
        sdims=sdims, schan=schan, img=image, chdim=2
    )
    d.addPairwiseEnergy(pairwise_bilateral, compat=10)

    # Optional Gaussian term
    # d.addPairwiseGaussian(sxy=(3, 3), compat=3)

    # --- Inference ---
    Q = np.array(d.inference(n_iters))
    probabilities = Q.reshape((n_classes, H, W))

    refined_segmentation = np.argmax(probabilities, axis=0).astype(np.uint8)

    # --- Uncertainty: entropy ---
    probabilities_clamped = np.clip(probabilities, epsilon, 1 - epsilon)
    refined_uncertainty = -np.sum(probabilities_clamped * np.log(probabilities_clamped), axis=0)

    if n_classes == 2:
        return probabilities[1], refined_segmentation, refined_uncertainty
    else:
        return probabilities, refined_segmentation, refined_uncertainty

def mc_dropout_inference(model: universeg, image_tensor: torch.Tensor, 
                        support_images: torch.Tensor, support_masks: torch.Tensor, 
                        num_samples: int, probability: float, activation='sigmoid'):
    mc_dropout_module = MCDropout(model=model, p=probability)
    mc_dropout_module.enable(
        layer_types=[nn.Linear, nn.Conv2d],
        ignore_type_layers=[nn.GELU, nn.ReLU, nn.LayerNorm, nn.LayerNorm, nn.Embedding],
    )

    all_probs = []
    masks_list = []
    images = [image_tensor.cpu().numpy()] * num_samples

    for _ in range(num_samples):
        prob_map, _ = universeg_inference(model=model, image_tensor=image_tensor, 
                                      support_images=support_images, support_masks=support_masks)
        all_probs.append(prob_map)
        masks_list.append((prob_map > 0.5).astype(np.uint8))

    mc_dropout_module.remove()

    if not all_probs:
         H, W = image_tensor.shape[-2:]
         return images, np.zeros((num_samples, H, W), dtype=np.uint8), np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    all_probs = np.stack(all_probs, axis=0)
    mean_probs = np.mean(all_probs, axis=0)
    EPSILON = np.finfo(np.float32).eps  # ≈1.192e-07
    mean_probs_clamped = np.clip(mean_probs, EPSILON, 1.0 - EPSILON)
    entropy_map = -(mean_probs_clamped * np.log(mean_probs_clamped) +
                  (1 - mean_probs_clamped) * np.log(1 - mean_probs_clamped))

    return images, np.array(masks_list), mean_probs, entropy_map


def tta_inference(model: universeg, image_tensor: torch.Tensor, support_images: torch.Tensor, 
                 support_masks: torch.Tensor, device: str, activation: str = "sigmoid"):
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(), # Optional
        tta.Rotate90(angles=[0, 90, 180, 270]), # Optional
        # tta.Scale(scales=[0.8, 1, 1.2]), # Use cautiously with SAM's fixed embed size
    ])

    tta_predictions_prob = []
    augmented_images = []
    masks_list = []

    image_tensor_device = image_tensor.to(device)
    # support_images_device = support_images.to(device)
    # support_masks_device = support_masks.to(device)

    for transform in transforms:
        augmented_image_tensor = transform.augment_image(image_tensor_device)
        aug_img_np = augmented_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        augmented_images.append(aug_img_np)

        prob_map, _ = universeg_inference(model=model, image_tensor=augmented_image_tensor,
                                       support_images=support_images, 
                                       support_masks=support_masks)

        prob_map_tensor = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0).to(device)
        deaugmented_prob_map_tensor = transform.deaugment_mask(prob_map_tensor)
        deaugmented_prob_map = deaugmented_prob_map_tensor.squeeze(0).squeeze(0).cpu().numpy()
        tta_predictions_prob.append(deaugmented_prob_map)
        masks_list.append((deaugmented_prob_map > 0.5).astype(np.uint8))

    if not tta_predictions_prob:
         H, W = image_tensor.shape[-2:]
         return augmented_images, np.zeros((len(transforms), H, W), dtype=np.uint8), np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    tta_predictions_prob = np.stack(tta_predictions_prob, axis=0)
    mean_probs = np.mean(tta_predictions_prob, axis=0)
    EPSILON = np.finfo(np.float32).eps  # ≈1.192e-07
    mean_probs_clamped = np.clip(mean_probs, EPSILON, 1.0 - EPSILON)
    entropy_map = -(mean_probs_clamped * np.log(mean_probs_clamped) +
                  (1 - mean_probs_clamped) * np.log(1 - mean_probs_clamped))

    return augmented_images, np.array(masks_list), mean_probs, entropy_map


def noisy_inference(model: universeg, noisy_model: NoisyInference, 
                   support_images: torch.Tensor, support_masks: torch.Tensor):
    noisy_samples = noisy_model.generate_noisy_samples()
    all_probs = []
    noisy_images = []
    masks_list = []

    for noisy_tensor in noisy_samples:
        noisy_img_np = noisy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        noisy_images.append(noisy_img_np)

        prob_map, _ = universeg_inference(model=model, image_tensor=noisy_tensor,
                                       support_images=support_images, 
                                       support_masks=support_masks)
        all_probs.append(prob_map)
        masks_list.append((prob_map > 0.5).astype(np.uint8))

    if not all_probs:
        try:
             H, W = noisy_samples[0].shape[-2:]
        except IndexError:
             H, W = 256, 256
        return noisy_images, np.zeros((len(noisy_samples), H, W), dtype=np.uint8), np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    all_probs = np.stack(all_probs, axis=0)
    mean_probs = np.mean(all_probs, axis=0)
    EPSILON = np.finfo(np.float32).eps  # ≈1.192e-07
    mean_probs_clamped = np.clip(mean_probs, EPSILON, 1.0 - EPSILON)
    entropy_map = -(mean_probs_clamped * np.log(mean_probs_clamped) +
                  (1 - mean_probs_clamped) * np.log(1 - mean_probs_clamped))

    return noisy_images, np.array(masks_list), mean_probs, entropy_map


# [save_outputs function remains largely unchanged, but ensure it handles numpy arrays correctly]
# ... (Keep original save_outputs function) ...
def save_outputs(images, masks, mean_prediction, uncertainty, mask_prediction, method_dir, sample_idx):
    """ Save outputs in an organized directory structure. """
    sample_dir = method_dir # Use the method-specific dir directly
    os.makedirs(sample_dir, exist_ok=True)

    images_dir = os.path.join(sample_dir, "images")
    predictions_dir = os.path.join(sample_dir, "predictions")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Save individual images (could be original, augmented, or noisy)
    # Ensure images are numpy arrays before saving
    if isinstance(images, list) and len(images) > 0:
         for i, img in enumerate(images):
              img_np = img if isinstance(img, np.ndarray) else np.asarray(img)
              # Handle tensor format if necessary (e.g., from MC dropout list)
              if img_np.ndim == 4 and img_np.shape[0] == 1: # Batch dim
                    img_np = img_np.squeeze(0)
              if img_np.shape[0] == 3: # Channel dim first
                    img_np = np.transpose(img_np, (1, 2, 0)) # HWC
              if img_np.dtype == np.float32 or img_np.dtype == np.float64: # Scale if needed
                   if img_np.min() >= 0 and img_np.max() <= 1:
                        img_np = (img_np * 255).astype(np.uint8)

              save_image(img_np, os.path.join(images_dir, f"image_{i}.png"))
    elif isinstance(images, np.ndarray): # Handle single image case
         save_image(images, os.path.join(images_dir, f"image_0.png"))


    # Save individual prediction masks
    if isinstance(masks, np.ndarray):
        # If it's a stack of masks (N, H, W)
        if masks.ndim == 3:
             for i, mask in enumerate(masks):
                  save_image(mask, os.path.join(predictions_dir, f"prediction_{i}.png"))
        elif masks.ndim == 2: # Single mask
             save_image(masks, os.path.join(predictions_dir, f"prediction_0.png"))
    elif isinstance(masks, list) and len(masks) > 0: # List of masks
        for i, mask in enumerate(masks):
             save_image(np.asarray(mask), os.path.join(predictions_dir, f"prediction_{i}.png"))

    # Save mean prediction probability, uncertainty, and the thresholded mean mask
    save_image(np.asarray(mean_prediction), os.path.join(sample_dir, "mean_prediction.png"))
    # Ensure uncertainty is scaled appropriately for saving as image
    uncertainty_np = np.asarray(uncertainty)
    uncertainty_scaled = (uncertainty_np - np.min(uncertainty_np)) / (np.ptp(uncertainty_np) + 1e-8)
    save_image(uncertainty_scaled, os.path.join(sample_dir, "uncertainty.png"))
    save_image(np.asarray(mask_prediction), os.path.join(sample_dir, "mean_mask_prediction.png"))


# [certainty_score function remains unchanged]
def certainty_score(uncertainty_map, ground_truth):
    uncertainty_map = np.squeeze(np.asarray(uncertainty_map))
    ground_truth = np.squeeze(np.asarray(ground_truth))
    # Ensure GT is boolean/binary
    ground_truth = (ground_truth > 0.5).astype(bool)

    if np.count_nonzero(ground_truth) == 0:
        return np.nan # Avoid division by zero if GT mask is empty

    # Lower uncertainty means higher certainty
    certainty_map = 1.0 - uncertainty_map # Convert uncertainty to certainty
    # Clamp certainty to [0, 1] just in case
    certainty_map = np.clip(certainty_map, 0.0, 1.0)

    certainty_in_gt = certainty_map[ground_truth]

    # Return the average certainty within the ground truth region
    return np.mean(certainty_in_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation pipeline with YAML configuration and metrics")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    config_paths = config["paths"]
    config_inference = config["inference"]
    # ---------------------------------
    config_fusion = config["fusion"]
    config_crf = config["crf"]
    config_noisy = config["noisy"]
    config_mc_dropout = config.get("mc_dropout", {}) # Get MC dropout params

    SAVE_PATH = config_paths["save_path"]
    # image_dir = config_paths["image_path"] # Not used directly if root_path is used
    # mask_dir = config_paths["mask_path"] # Not used directly if root_path is used
    root_dir = config_paths["root_path"]


    pairs = recover_image_mask_pairs(root_dir=root_dir)

    if not pairs:
        print(f"Error: No image-mask pairs found in {root_dir}. Please check the path and data structure.")
        exit()
    
    subset_size = config.get("subset", {}).get("subset_size", len(pairs))
    context_set_size = config.get("subset", {}).get("size_context_set", 4)
    context_set = pairs
    if subset_size and subset_size > 0:
        context_set = pairs[:context_set_size]
        pairs = pairs[context_set_size:subset_size+context_set_size]
        print(f"Processing subset of {subset_size} samples. With a context set size of {context_set_size}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load UniVerSeg Model ---
    print("Loading UniVerSeg model...")
    model = universeg(pretrained=True)

    try:
        # Instantiate the SAM model based on the type
        model.to(device=device)
        model.eval() # Set model to evaluation mode

        print(f"UniVerSeg model loaded successfully.")
    except Exception as e:
        print(f"Error loading UniVerSeg model: {e}")
        exit()

    #### Load Context Set
    print("Loading context set")
    support_images, support_masks = [], []
    for idx, (img_path, mask_path) in enumerate(context_set):
        # Load image and mask
        image_pil = PIL.Image.open(img_path).convert("RGB")  # UniVerSeg requires Gray image
        gt_mask_pil = PIL.Image.open(mask_path).convert("L")

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # Converts to [0, 1] tensor (1, H, W)
            torchvision.transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        ])

        # Process image (already shape (1, H, W) after ToTensor)
        image_tensor = transform(image_pil).to(device)  # (1, H, W)
        image_tensor = image_tensor.mean(dim=0, keepdim=True)  # (1, H, W)
        support_images.append(image_tensor.unsqueeze(0).unsqueeze(0))  # Add batch dim -> (1, 1, H, W)

        # Process mask (ensure same shape)
        gt_mask_tensor = transform(gt_mask_pil)  # (1, H, W)
        gt_mask_binary = (gt_mask_tensor > 0.5).float()  # Convert to binary 0/1 and keep as tensor
        support_masks.append(gt_mask_binary.unsqueeze(0).unsqueeze(0))  # Add batch dim -> (1, 1, H, W)

    # Stack along S dimension to get (1, S, 1, H, W)
    support_images = torch.cat(support_images, dim=1).to(device)  # (1, S, 1, H, W)
    support_masks = torch.cat(support_masks, dim=1).to(device)   # (1, S, 1, H, W)

    print("Context set loaded")
    print(f"Support images shape: {support_images.shape}")
    print(f"Support masks shape: {support_masks.shape}")
    
    # -------------------------

    overall_metrics = {}

    for idx, (img_path, mask_path) in enumerate(pairs):
        print(f"\n--- Processing Sample {idx+1}/{len(pairs)}: {os.path.basename(img_path)} ---")
        try:
            # Load image and mask
            image_pil = PIL.Image.open(img_path).convert("L") # UniVerSeg requires Gray image
            gt_mask_pil = PIL.Image.open(mask_path).convert("L")
            gt_mask_np = np.array(gt_mask_pil) > 0.5

            # --- Preprocessing ---
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), # Converts to [0, 1] tensor C, H, W
                # Required a size 128,128 for UniVerSeg
                torchvision.transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                # Add any other minimal preprocessing if needed, but avoid resizing here
            ])
            image_tensor = transform(image_pil).to(device) # Add batch dim, send to device
            image_tensor = image_tensor.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, H, W)
            gt_mask_tensor = transform(gt_mask_pil) # No batch dim needed, keep on CPU usually
            gt_mask_np = gt_mask_tensor.cpu().numpy().squeeze() # Squeeze channel dim if present
            gt_mask_np = (gt_mask_np > 0.5).astype(np.uint8) # Ensure binary 0/1 numpy


            # Create a dedicated directory for this sample
            sample_root_dir = os.path.join(SAVE_PATH, f"sample_{idx}")
            os.makedirs(sample_root_dir, exist_ok=True)

            # Save original image and ground truth
            save_image(image_pil, os.path.join(sample_root_dir, "original_image.png"))
            save_image(gt_mask_np, os.path.join(sample_root_dir, "ground_truth_mask.png"))

            sample_metrics = {} # Metrics for this specific sample

            # ----- Normal Inference -----
            print("Running Normal Inference...")
            normal_prob, normal_mask = normal_inference(model=model, image_tensor=image_tensor, 
                                                        support_images=support_images, support_masks=support_masks)

            normal_iou = compute_iou(normal_mask, gt_mask_np)
            normal_dice = compute_dice(normal_mask, gt_mask_np)
            normal_metrics_dict = compute_metrics(normal_prob, gt_mask_np)
            # Uncertainty for normal inference is just 1 - probability
            p_normal = normal_prob
            p_other = 1.0 - normal_prob

            # Stack the probabilities along a new axis to form the distributions
            # If normal_prob is shape (H, W), prob_distribution will be (H, W, 2)
            prob_distribution = np.stack([p_normal, p_other], axis=-1)
            def numpy_entropy(p, axis=-1, base=np.e):
                """Calculates entropy using numpy."""
                EPSILON = np.finfo(np.float32).eps
                p = np.clip(p, EPSILON, 1.0 - EPSILON)
                log_p = np.log(p)
                return -np.sum(p * log_p, axis=axis)
            
            normal_uncertainty = numpy_entropy(prob_distribution, axis=-1, base=np.e)
            normal_certainty = certainty_score(normal_uncertainty, gt_mask_np)
            sample_metrics["normal"] = {
                "iou": normal_iou, "dice": normal_dice, "certainty": normal_certainty,
                **normal_metrics_dict
            }

            # Save normal inference results
            normal_dir = os.path.join(sample_root_dir, "original")
            os.makedirs(normal_dir, exist_ok=True)
            save_image(normal_prob, os.path.join(normal_dir, "probability.png"))
            save_image(normal_mask, os.path.join(normal_dir, "mask.png"))
            save_image(normal_uncertainty, os.path.join(normal_dir, "uncertainty.png"))
            print(f"  Normal - IoU: {normal_iou:.4f}, Dice: {normal_dice:.4f}, Certainty: {normal_certainty:.4f}")
        
            # ----- MC Dropout Inference (MedSAM) -----
            print("Running MC Dropout Inference...")
            num_samples_mc = config_inference.get("num_samples", 10)

            # Pass the underlying SAM model to MCDropout, and predictor for inference calls
            mc_images_ref, mc_masks, mc_mean_prediction, mc_entropy = mc_dropout_inference(
                model=model, image_tensor=image_tensor, support_images=support_images, support_masks=support_masks,
                probability=config_mc_dropout.get("p", 0.1),
                num_samples=num_samples_mc # 'activation' ignored
            )
            mc_mask_pred = (mc_mean_prediction > 0.5).astype(np.uint8)
            mc_iou = compute_iou(mc_mask_pred, gt_mask_np)
            mc_dice = compute_dice(mc_mask_pred, gt_mask_np)
            mc_metrics_dict = compute_metrics(mc_mean_prediction, gt_mask_np)
            mc_certainty = certainty_score(mc_entropy, gt_mask_np) # Use entropy as uncertainty
            sample_metrics["mc_dropout"] = {
                "iou": mc_iou, "dice": mc_dice, "certainty": mc_certainty,
                **mc_metrics_dict
            }

            # Save MC dropout results
            mc_dropout_dir = os.path.join(sample_root_dir, "mc_dropout")
            save_outputs(mc_images_ref, mc_masks, mc_mean_prediction, mc_entropy, mc_mask_pred, mc_dropout_dir, idx)
            print(f"  MC Dropout - IoU: {mc_iou:.4f}, Dice: {mc_dice:.4f}, Certainty: {mc_certainty:.4f}")

        


            # ----- TTA Inference -----
            print("Running TTA Inference...")
            tta_images, tta_masks, tta_mean_prediction, tta_entropy = tta_inference(
                model=model, image_tensor=image_tensor, support_images=support_images, support_masks=support_masks, device=device # 'activation' ignored
            )
            tta_mask_pred = (tta_mean_prediction > 0.5).astype(np.uint8)
            tta_iou = compute_iou(tta_mask_pred, gt_mask_np)
            tta_dice = compute_dice(tta_mask_pred, gt_mask_np)
            tta_metrics_dict = compute_metrics(tta_mean_prediction, gt_mask_np)
            tta_certainty = certainty_score(tta_entropy, gt_mask_np) # Use entropy as uncertainty
            sample_metrics["tta"] = {
                "iou": tta_iou, "dice": tta_dice, "certainty": tta_certainty,
                **tta_metrics_dict
            }

            # Save TTA results
            tta_dir = os.path.join(sample_root_dir, "tta")
            save_outputs(tta_images, tta_masks, tta_mean_prediction, tta_entropy, tta_mask_pred, tta_dir, idx)
            print(f"  TTA - IoU: {tta_iou:.4f}, Dice: {tta_dice:.4f}, Certainty: {tta_certainty:.4f}")


            # ----- Noisy Inference (MedSAM) -----
            print("Running Noisy Inference...")
            num_samples_noisy = config_inference.get("num_samples", 10)
            noisy_perturber = NoisyInference(
                image=image_tensor, # Pass the tensor here
                N_SAMPLES=num_samples_noisy,
                noise_std=config_noisy.get("noise_std", 0.1)
            )
            # Pass the predictor and the noisy_perturber object
            noise_images, noise_masks, noise_mean_prediction, noise_entropy = noisy_inference(
                model=model, noisy_model=noisy_perturber, support_images=support_images, support_masks=support_masks # 'activation' ignored
            )
            noise_mask_pred = (noise_mean_prediction > 0.5).astype(np.uint8)
            noise_iou = compute_iou(noise_mask_pred, gt_mask_np)
            noise_dice = compute_dice(noise_mask_pred, gt_mask_np)
            noise_metrics_dict = compute_metrics(noise_mean_prediction, gt_mask_np)
            noise_certainty = certainty_score(noise_entropy, gt_mask_np) # Use entropy as uncertainty
            sample_metrics["noisy"] = {
                "iou": noise_iou, "dice": noise_dice, "certainty": noise_certainty,
                **noise_metrics_dict
            }

            # Save noisy inference results
            noisy_dir = os.path.join(sample_root_dir, "noisy")
            save_outputs(noise_images, noise_masks, noise_mean_prediction, noise_entropy, noise_mask_pred, noisy_dir, idx)
            print(f"  Noisy - IoU: {noise_iou:.4f}, Dice: {noise_dice:.4f}, Certainty: {noise_certainty:.4f}")


            # ----- Fusion of Uncertainty and Segmentation -----
            print("Running Fusion...")
            # Use entropy maps from MC, TTA, Noisy as the uncertainty inputs
            consensus_prob, consensus_uncertainty, consensus_mask_dynamic = weighted_average_with_uncertainty(
                mc_mean_prediction, mc_entropy,
                tta_mean_prediction, tta_entropy,
                noise_mean_prediction, noise_entropy,
                weighting_method=config_fusion.get("weighting_method", "inverse"),
                beta=config_fusion.get("beta", 1.0),
                alpha=config_fusion.get("alpha", 1.0),
                threshold_method=config_fusion.get("threshold_method", "otsu"), # Use dynamic threshold on consensus_prob
                percentile=config_fusion.get("percentile", 50),
                k=config_fusion.get("k", 0.5),
                epsilon=float(config_fusion.get("epsilon", 1e-6))
            )
            # Also compute simple threshold mask for comparison if needed
            fusion_mask_naive = (consensus_prob > 0.5).astype(np.uint8)

            # Evaluate using the dynamically thresholded mask from the function
            fusion_iou = compute_iou(consensus_mask_dynamic, gt_mask_np)
            fusion_dice = compute_dice(consensus_mask_dynamic, gt_mask_np)
            fusion_metrics_dict = compute_metrics(consensus_prob, gt_mask_np) # Metrics based on probability map
            fusion_certainty = certainty_score(consensus_uncertainty, gt_mask_np) # Use fused uncertainty
            sample_metrics["fusion"] = {
                "iou": fusion_iou, "dice": fusion_dice, "certainty": fusion_certainty,
                **fusion_metrics_dict
            }

            # Save fusion results
            fusion_dir = os.path.join(sample_root_dir, "fusion")
            os.makedirs(fusion_dir, exist_ok=True)
            save_image(consensus_prob, os.path.join(fusion_dir, "probability.png"))
            save_image(consensus_mask_dynamic, os.path.join(fusion_dir, "mask_dynamic_thresh.png")) # Save dynamic mask
            save_image(fusion_mask_naive, os.path.join(fusion_dir, "mask_naive_thresh.png")) # Save naive mask
            save_image(consensus_uncertainty, os.path.join(fusion_dir, "uncertainty.png"))
            print(f"  Fusion (Dynamic Thresh) - IoU: {fusion_iou:.4f}, Dice: {fusion_dice:.4f}, Certainty: {fusion_certainty:.4f}")


            # ----- CRF Refinement -----
            # Apply CRF to the fused probability and uncertainty maps
            print("Running CRF Refinement...")
            # refine_with_crf_uncertainty expects binary prob map (prob of class 1)
            # and corresponding uncertainty map.
            # image_tensor needs to be passed for pairwise term calculation.
            crf_prob_class1, final_segmentation, final_uncertainty = refine_with_crf_uncertainty(
                image_tensor, consensus_prob, consensus_uncertainty, # Pass fused results
                sdims=tuple(config_crf.get("sdims", (5, 5))),
                schan=tuple(config_crf.get("schan", (5, ))),
                n_iters=config_crf.get("n_iters", 5),
                epsilon=float(config_crf.get("epsilon", 1e-8))
            )
            # Evaluate the final CRF mask
            crf_iou = compute_iou(final_segmentation, gt_mask_np)
            crf_dice = compute_dice(final_segmentation, gt_mask_np)
            # Use the probability map output by CRF for other metrics
            crf_metrics_dict = compute_metrics(crf_prob_class1, gt_mask_np)
            crf_certainty = certainty_score(final_uncertainty, gt_mask_np) # Use CRF's uncertainty output
            sample_metrics["crf"] = {
                "iou": crf_iou, "dice": crf_dice, "certainty": crf_certainty,
                **crf_metrics_dict
            }

            # Save CRF results
            crf_dir = os.path.join(sample_root_dir, "refined")
            os.makedirs(crf_dir, exist_ok=True)
            save_image(crf_prob_class1, os.path.join(crf_dir, "probability.png")) # Prob of class 1 after CRF
            save_image(final_segmentation, os.path.join(crf_dir, "mask.png"))
            save_image(final_uncertainty, os.path.join(crf_dir, "uncertainty.png"))
            print(f"  CRF Refinement - IoU: {crf_iou:.4f}, Dice: {crf_dice:.4f}, Certainty: {crf_certainty:.4f}")


            # Store metrics for this sample
            overall_metrics[f"sample_{idx}"] = sample_metrics

        except FileNotFoundError:
            print(f"Error: Image or mask not found for sample {idx}: {img_path} or {mask_path}")
            continue # Skip to next sample
        except ImportError as e:
             print(f"Import Error: {e}. Make sure all required packages are installed.")
             break # Stop processing
        except Exception as e:
            print(f"Error processing sample {idx} ({os.path.basename(img_path)}): {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            # Store NaN for this sample's metrics to avoid breaking aggregation
            overall_metrics[f"sample_{idx}"] = {method: {metric: np.nan for metric in metrics} for method in methods}
            continue # Skip to next sample


    # --- Aggregation and Visualization ---
    print("\n--- Aggregating and Saving Results ---")
    if not overall_metrics:
         print("No samples were processed successfully. Skipping aggregation.")
         exit()


    visualization_dir = os.path.join(SAVE_PATH, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    methods = ['normal', 'mc_dropout', 'tta', 'noisy', 'fusion', 'crf']
    metrics = ['iou', 'dice', 'nll', 'ece', 'brier', 'accuracy', 'precision', 'recall', 'certainty']

    # Use pandas for easier aggregation and handling of missing data (NaN)
    # Create a flat dictionary first { ('sample_id', 'method', 'metric'): value }
    flat_metrics = {}
    for sample_id, methods_data in overall_metrics.items():
         for method, metrics_data in methods_data.items():
              if method in methods: # Only include expected methods
                   for metric, value in metrics_data.items():
                        if metric in metrics: # Only include expected metrics
                             flat_metrics[(sample_id, method, metric)] = value

    # Convert to MultiIndex DataFrame
    if not flat_metrics:
         print("No valid metric data collected. Cannot create summary.")
         exit()

    multi_index = pd.MultiIndex.from_tuples(flat_metrics.keys(), names=['sample', 'method', 'metric'])
    metrics_df = pd.Series(flat_metrics, index=multi_index)

    # Unstack to get methods as columns, metrics as lower index
    metrics_table = metrics_df.unstack(level='method')
    # Further unstack to get metrics as columns
    detailed_table = metrics_table.unstack(level='metric')


    # Save detailed table (Samples x (Method_Metric))
    detailed_csv_path = os.path.join(visualization_dir, 'detailed_metrics_per_sample.csv')
    try:
        detailed_table.to_csv(detailed_csv_path)
        print(f"Detailed metrics per sample saved to {detailed_csv_path}")
    except Exception as e:
        print(f"Error saving detailed CSV: {e}")


    # Calculate mean across samples for the summary
    # Group by method and metric, then calculate mean
    mean_results = metrics_df.groupby(level=['method', 'metric']).mean()
    mean_summary_table = mean_results.unstack(level='metric') # Metrics as columns, methods as rows

    # Reindex to ensure all methods/metrics are present, fill missing with NaN
    mean_summary_table = mean_summary_table.reindex(index=methods, columns=metrics)


    # Save summary table
    summary_csv_path = os.path.join(visualization_dir, 'metrics_summary_mean.csv')
    try:
        mean_summary_table.to_csv(summary_csv_path)
        print(f"Mean metrics summary saved to {summary_csv_path}")
        print("\nMean Metrics Summary:")
        print(mean_summary_table)
    except Exception as e:
        print(f"Error saving summary CSV: {e}")


    # --- Plotting ---
    # Plot mean metrics using the summary table
    try:
        plt.style.use('ggplot') # Use a style for nicer plots
        num_metrics = len(metrics)
        num_cols = 3
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows), squeeze=False)
        axes = axes.flatten() # Flatten to easily iterate

        colors = plt.cm.get_cmap('tab10', len(methods)) # Get distinct colors

        for i, metric in enumerate(metrics):
            if metric in mean_summary_table.columns:
                ax = axes[i]
                metric_data = mean_summary_table[metric].reindex(methods) # Ensure correct order
                bars = ax.bar(metric_data.index, metric_data.values, color=[colors(j) for j in range(len(methods))])
                ax.set_title(metric.upper(), fontsize=14)
                ax.set_ylabel("Score", fontsize=12)
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                         ax.annotate(f'{height:.3f}',
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3), # 3 points vertical offset
                                      textcoords="offset points",
                                      ha='center', va='bottom', fontsize=9)
            else:
                 axes[i].set_title(f"{metric.upper()} (No Data)", fontsize=14)
                 axes[i].axis('off') # Hide axes if no data

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        fig.suptitle("Mean Performance Metrics Comparison", fontsize=18, y=0.99)

        # Add a single legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors(j)) for j in range(len(methods))]
        fig.legend(handles, methods, loc='upper right', bbox_to_anchor=(0.98, 0.92), fontsize=12, title="Methods")


        plot_path = os.path.join(visualization_dir, 'mean_metrics_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Mean metrics plot saved to {plot_path}")

    except Exception as e:
        print(f"Error generating mean metrics plot: {e}")
        import traceback
        traceback.print_exc()


    # --- Box Plots for Distribution ---
    try:
        # Use the detailed table reshaped: Samples x Methods for each metric
        metric_boxplot_data = metrics_df.unstack(level='method') # Index: (sample, metric), Columns: method

        num_metrics = len(metrics)
        num_cols = 3
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows), squeeze=False)
        axes = axes.flatten()

        colors = plt.cm.get_cmap('tab10', len(methods))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            # Select data for the current metric, drop samples with NaN for this metric
            if metric in metric_boxplot_data.index.get_level_values('metric'):
                 data_for_metric = metric_boxplot_data.loc[(slice(None), metric), :].droplevel('metric').reindex(columns=methods) # Order columns
                 # Convert to list of arrays for boxplot, handling potential NaNs per method
                 plot_data = [data_for_metric[method].dropna().values for method in methods]
                 valid_methods = [methods[k] for k, d in enumerate(plot_data) if len(d) > 0] # Methods with data
                 plot_data_filtered = [d for d in plot_data if len(d) > 0] # Data for valid methods
                 method_colors = [colors(methods.index(m)) for m in valid_methods] # Colors for valid methods

                 if plot_data_filtered:
                      bp = ax.boxplot(plot_data_filtered, labels=valid_methods, patch_artist=True, vert=True, showfliers=False) # Hide outliers for cleaner look maybe

                      for patch, color in zip(bp['boxes'], method_colors):
                           patch.set_facecolor(color)
                           patch.set_alpha(0.7)
                      for median in bp['medians']:
                           median.set_color('black')

                      ax.set_title(metric.upper(), fontsize=14)
                      ax.set_ylabel("Score Distribution", fontsize=12)
                      ax.tick_params(axis='x', rotation=45, labelsize=10)
                      ax.tick_params(axis='y', labelsize=10)
                      ax.grid(axis='y', linestyle='--', alpha=0.7)
                 else:
                      ax.set_title(f"{metric.upper()} (No Data)", fontsize=14)
                      ax.axis('off')
            else:
                 ax.set_title(f"{metric.upper()} (No Data)", fontsize=14)
                 ax.axis('off')


        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
             fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Metric Score Distribution Across Samples", fontsize=18, y=0.99)

        # Add a single legend (optional, colors identify methods in each plot)
        # handles = [plt.Rectangle((0,0),1,1, color=colors(j)) for j in range(len(methods))]
        # fig.legend(handles, methods, loc='upper right', bbox_to_anchor=(0.98, 0.92), fontsize=12, title="Methods")

        boxplot_path = os.path.join(visualization_dir, 'metrics_distribution_boxplots.png')
        plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Metrics distribution boxplot saved to {boxplot_path}")

    except Exception as e:
        print(f"Error generating box plots: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Pipeline Finished ---")