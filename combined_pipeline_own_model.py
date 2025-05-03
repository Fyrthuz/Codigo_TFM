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

# Import custom modules
from mc_dropout import MCDropout
from noise_inference import NoisyInference

# pydensecrf imports
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

# ----------------- Helper functions -----------------

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

def save_image(array, path):
    # [Unchanged from original code]
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.squeeze(array)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    if array.ndim == 2:
        array = (array * 255).astype(np.uint8)
        image = PIL.Image.fromarray(array, mode='L')
        image.save(path)
        return
    if array.ndim == 3:
        if array.shape[0] in [1, 3]:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)
        if array.shape[2] == 1:
            array = array[:,:,0]
            image = PIL.Image.fromarray(array, mode='L')
        else:
            image = PIL.Image.fromarray(array)
        image.save(path)
        return
    raise ValueError(f"Unsupported array shape: {array.shape}")

def compute_iou(pred, target):
    # [Unchanged from original code]
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    intersection_sum = intersection.sum()
    union_sum = union.sum()
    if union_sum == 0:
        return 1.0 if intersection_sum == 0 else 0.0
    return intersection_sum / union_sum

def compute_dice(pred, target):
    # [Unchanged from original code]
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target)
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum + target_sum == 0:
        return 1.0
    return (2 * intersection.sum()) / (pred_sum + target_sum)

def compute_metrics(prob, gt_mask, epsilon=1e-8):
    prob_flat = prob.flatten()
    gt_flat = gt_mask.flatten().astype(np.int64)
    prob_flat = np.clip(prob_flat, epsilon, 1 - epsilon)

    # Negative Log Likelihood
    nll = -np.mean(gt_flat * np.log(prob_flat + epsilon) + (1 - gt_flat) * np.log(1 - prob_flat + epsilon))
    
    # Brier Score
    brier = np.mean((prob_flat - gt_flat) ** 2)
    
    # Predicted mask
    pred_mask = (prob_flat > 0.5).astype(np.int64)
    
    # Confusion matrix components
    tp = np.sum((pred_mask == 1) & (gt_flat == 1))
    fp = np.sum((pred_mask == 1) & (gt_flat == 0))
    tn = np.sum((pred_mask == 0) & (gt_flat == 0))
    fn = np.sum((pred_mask == 0) & (gt_flat == 1))
    
    # Accuracy, Precision, Recall
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, 11)
    bin_indices = np.digitize(prob_flat, bin_edges, right=True)
    ece = 0.0
    for i in range(1, 11):
        mask = (bin_indices == i)
        bin_size = np.sum(mask)
        if bin_size == 0:
            continue
        conf = np.mean(prob_flat[mask])
        acc = np.mean((pred_mask[mask] == gt_flat[mask]).astype(float))
        ece += np.abs(acc - conf) * bin_size
    ece /= len(prob_flat)
    
    return {
        'nll': nll,
        'ece': ece,
        'brier': brier,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

def normal_inference(model, image, activation='sigmoid'):
    # [Unchanged from original code]
    with torch.no_grad():
        output = model(image)
    if activation == "sigmoid":
        prob = output.squeeze(0).cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8)
    elif activation == "softmax":
        prob = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
        mask = np.argmax(prob, axis=0)
    else:
        raise ValueError("activation must be 'sigmoid' or 'softmax'")
    return prob, mask

def dynamic_threshold_multiclass(prob_maps, method="otsu", percentile=50, k=0.5, epsilon=1e-6):
    # [Unchanged from original code]
    if prob_maps.ndim == 2:
        if method.lower() == "otsu":
            threshold = threshold_otsu(prob_maps)
        elif method.lower() == "percentile":
            threshold = np.percentile(prob_maps, percentile)
        elif method.lower() == "mean_std":
            mean_val = np.mean(prob_maps)
            std_val = np.std(prob_maps)
            threshold = mean_val + k * std_val
        else:
            raise ValueError("Unknown method. Choose from 'otsu', 'percentile', or 'mean_std'.")
        return np.array([threshold])
    elif prob_maps.ndim == 3:
        num_classes = prob_maps.shape[0]
        thresholds = np.zeros(num_classes)
        for c in range(num_classes):
            class_probs = prob_maps[c]
            if method.lower() == "otsu":
                thresholds[c] = threshold_otsu(class_probs)
            elif method.lower() == "percentile":
                thresholds[c] = np.percentile(class_probs, percentile)
            elif method.lower() == "mean_std":
                mean_val = np.mean(class_probs)
                std_val = np.std(class_probs)
                thresholds[c] = mean_val + k * std_val
            else:
                raise ValueError("Unknown method. Choose from 'otsu', 'percentile', or 'mean_std'.")
        return thresholds
    else:
        raise ValueError("prob_maps must be either a 2D (binary) or 3D (multiclass) array.")

def weighted_average_with_uncertainty(mc_mean, mc_uncert, tta_mean, tta_uncert, noise_mean, noise_uncert,
                                      weighting_method="inverse", beta=1.0, alpha=1.0,
                                      threshold_method="otsu", percentile=50, k=0.5, epsilon=1.e-6):
    # [Unchanged from original code]
    mc_mean = np.squeeze(mc_mean)
    tta_mean = np.squeeze(tta_mean)
    noise_mean = np.squeeze(noise_mean)
    mc_uncert = np.squeeze(mc_uncert)
    tta_uncert = np.squeeze(tta_uncert)
    noise_uncert = np.squeeze(noise_uncert)

    prob_maps = np.stack([mc_mean, tta_mean, noise_mean], axis=0)
    uncertainty_maps = np.stack([mc_uncert, tta_uncert, noise_uncert], axis=0)
    
    if weighting_method.lower() == "inverse":
        weights = 1.0 / (uncertainty_maps + epsilon)
    elif weighting_method.lower() == "exponential":
        weights = np.exp(-beta * uncertainty_maps)
    elif weighting_method.lower() == "powerlaw":
        weights = (1.0 - uncertainty_maps) ** alpha
    else:
        raise ValueError("Unsupported weighting method.")
    
    weights = weights / (np.sum(weights, axis=0, keepdims=True) + epsilon)
    
    consensus_prob = np.sum(prob_maps * weights, axis=0)
    consensus_uncertainty = np.sum(uncertainty_maps * weights, axis=0)

    if consensus_prob.ndim == 2:
        num_classes = 1
        H, W = consensus_prob.shape
    else:
        num_classes, H, W = consensus_prob.shape

    if threshold_method == 'naive':
        if num_classes == 1:
            consensus_mask = (consensus_prob > 0.5).astype(np.uint8)
        else:
            consensus_mask = np.argmax(consensus_prob, axis=0)

    else:
        thresholds = dynamic_threshold_multiclass(consensus_prob, method=threshold_method, percentile=percentile, k=k)

        if len(thresholds) == 1:
            class_masks = (consensus_prob > thresholds[0]).astype(np.uint8)
        else:
            class_masks = np.zeros((num_classes, H, W), dtype=np.uint8)
            for c in range(num_classes):
                class_masks[c] = (consensus_prob[c] > thresholds[c]).astype(np.uint8)
            
        if num_classes > 1:
            consensus_mask = np.argmax(class_masks, axis=0)
        else:
            consensus_mask = class_masks
    
    return consensus_prob, consensus_uncertainty, consensus_mask

def refine_with_crf_uncertainty(image, prob_map, uncertainty_map,
                                sdims=(5, 5), schan=(5, 5, 5),
                                n_iters=5, epsilon=1e-8):
    if not isinstance(image, np.ndarray):
        image = image.cpu().numpy()
    if image.ndim == 4 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)

    if prob_map.ndim == 2:
        prob_stack = np.stack([1 - prob_map, prob_map], axis=0)
        n_classes = 2
    elif prob_map.ndim == 3:
        prob_stack = prob_map
        n_classes = prob_map.shape[0]
    else:
        raise ValueError("prob_map must be 2D (binary) or 3D (multiclass)")
    H, W = prob_stack.shape[1:]
    
    d = dcrf.DenseCRF2D(W, H, n_classes)
    unary = unary_from_softmax(prob_stack)
    
    norm_uncert = (uncertainty_map - np.min(uncertainty_map)) / (np.ptp(uncertainty_map) + epsilon)
    norm_uncert_flat = norm_uncert.flatten()
    uniform_unary = np.ones((n_classes, H*W), dtype=np.float32) / n_classes
    adjusted_unary = (1 - norm_uncert_flat) * unary + norm_uncert_flat * uniform_unary
    
    d.setUnaryEnergy(adjusted_unary)
    pairwise_bilateral = create_pairwise_bilateral(sdims=sdims,
                                                     schan=schan,
                                                     img=image,
                                                     chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=10)
    
    Q = d.inference(n_iters)
    Q = np.array(Q)
    probabilities = Q.reshape(n_classes, H, W)
    refined_segmentation = np.argmax(probabilities, axis=0)
    refined_uncertainty = -np.sum(probabilities * np.log(probabilities + epsilon), axis=0)
    
    return probabilities, refined_segmentation, refined_uncertainty

def mc_dropout_inference(model, image, num_samples, activation='sigmoid'):
    # [Unchanged from original code]
    images = []
    masks_list = []
    
    with torch.no_grad():
        outputs = [model(image) for _ in range(num_samples)]
        images.extend([image.cpu().numpy() for _ in range(num_samples)])
    
    outputs = torch.stack(outputs)

    if activation == "softmax":
        softmax_preds = F.softmax(outputs, dim=1)
        mean_probs = softmax_preds.mean(dim=0).cpu().numpy()
        entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)
        masks_list = np.argmax(softmax_preds.cpu().numpy(), axis=1)
    elif activation == "sigmoid":
        sigmoid_preds = outputs
        mean_probs = sigmoid_preds.mean(dim=0).squeeze(0).cpu().numpy()
        entropy_map = -(mean_probs * np.log(mean_probs + 1e-8) + 
                        (1 - mean_probs) * np.log(1 - mean_probs + 1e-8))
        entropy_map = np.squeeze(entropy_map)
        masks_list = (sigmoid_preds.cpu().numpy() > 0.5).astype(np.uint8)
    else:
        raise ValueError("activation must be 'softmax' or 'sigmoid'")
    
    return images, masks_list, mean_probs, entropy_map

def tta_inference(model, image, device: str, activation: str = "sigmoid"):
    # [Unchanged from original code]
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.5, 1, 2]),
        tta.Multiply(factors=[0.8, 0.9, 1, 1.1, 1.2]),        
    ])
    
    tta_predictions = []
    augmented_images = []
    masks_list = []

    with torch.no_grad():
        for transform in transforms:
            augmented_image = transform.augment_image(image)
            augmented_images.append(augmented_image.cpu().numpy())
            output = model(augmented_image)
            output = transform.deaugment_mask(output)
            tta_predictions.append(output)

    tta_predictions = torch.stack(tta_predictions)

    if activation == "softmax":
        softmax_preds = F.softmax(tta_predictions, dim=1)
        mean_probs = softmax_preds.mean(dim=0).cpu().numpy()
        entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)
        masks_list = np.argmax(softmax_preds.cpu().numpy(), axis=1)
    elif activation == "sigmoid":
        sigmoid_preds = tta_predictions
        mean_probs = sigmoid_preds.mean(dim=0).cpu().numpy()
        entropy_map = -(mean_probs * np.log(mean_probs + 1e-8) + 
                        (1 - mean_probs) * np.log(1 - mean_probs + 1e-8))
        masks_list = (sigmoid_preds.cpu().numpy() > 0.5).astype(np.uint8)
    else:
        raise ValueError("activation must be 'softmax' or 'sigmoid'")
    
    return augmented_images, masks_list, mean_probs, entropy_map

def noisy_inference(noisy_model, model, activation="sigmoid"):
    # [Unchanged from original code]
    noisy_samples = noisy_model.generate_noisy_samples()
    all_probs = []
    noisy_images = []
    masks_list = []

    with torch.no_grad():
        for sample in noisy_samples:
            noisy_images.append(sample.cpu().numpy())
            output = model(sample)
            all_probs.append(output)

    all_probs = torch.stack(all_probs)

    if activation == "softmax":
        mean_probs = all_probs.mean(dim=0).cpu().numpy()
        entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)
        masks_list = np.argmax(all_probs.cpu().numpy(), axis=1)
    elif activation == "sigmoid":
        mean_probs = all_probs.mean(dim=0).cpu().numpy()
        entropy_map = -(mean_probs * np.log(mean_probs + 1e-8) + 
                        (1 - mean_probs) * np.log(1 - mean_probs + 1e-8))
        masks_list = (all_probs.cpu().numpy() > 0.5).astype(np.uint8)
    else:
        raise ValueError("activation must be 'softmax' or 'sigmoid'")
    
    return noisy_images, masks_list, mean_probs, entropy_map


# Modified save_outputs function to create a more organized directory structure
def save_outputs(images, masks, mean_prediction, uncertainty, mask_prediction, method_dir, sample_idx):
    """
    Save outputs in an organized directory structure.
    
    Directory structure:
    - method_dir/
      - sample_{sample_idx}/
        - images/        (input images or augmented versions)
        - predictions/   (individual prediction masks)
        - mean_prediction.png
        - uncertainty.png
    """
    # Create main sample directory for this method
    sample_dir = method_dir
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create subdirectories
    images_dir = os.path.join(sample_dir, "images")
    predictions_dir = os.path.join(sample_dir, "predictions")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    

    # Save individual images and masks
    if isinstance(images, list) and len(images) > 0:
        for i, img in enumerate(images):
            save_image(img, os.path.join(images_dir, f"image_{i}.png"))

    if isinstance(masks, np.ndarray):
        masks = [np.asarray(mask) for mask in masks]

    if isinstance(masks, list) and len(masks) > 0:
        for i, mask in enumerate(masks):
            save_image(mask, os.path.join(predictions_dir, f"prediction_{i}.png"))
    
    # Save mean prediction and uncertainty in the sample directory
    save_image(mean_prediction, os.path.join(sample_dir, "mean_prediction.png"))
    save_image(uncertainty, os.path.join(sample_dir, "uncertainty.png"))    
    save_image(mask_prediction, os.path.join(sample_dir, "mean_mask_prediction.png"))

def certainty_score(uncertainty_map, ground_truth, num_classes=2):
    uncertainty_map = np.squeeze(np.asarray(uncertainty_map))
    ground_truth = np.squeeze(np.asarray(ground_truth))
    # Ensure GT is boolean/binary
    ground_truth = (ground_truth > 0.5).astype(bool)

    if np.count_nonzero(ground_truth) == 0:
        return np.nan # Avoid division by zero if GT mask is empty

    max_entropy = np.log(num_classes)  # natural log
    normalized_certainty = 1.0 - (uncertainty_map / max_entropy)
    normalized_certainty = np.clip(normalized_certainty, 0.0, 1.0)

    certainty_in_gt = normalized_certainty[ground_truth]
    return np.mean(certainty_in_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation pipeline with YAML configuration and metrics")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    config_paths = config["paths"]
    config_inference = config["inference"]
    config_model = config["model"]
    config_fusion = config["fusion"]
    config_crf = config["crf"]
    config_noisy = config["noisy"]

    SAVE_PATH = config_paths["save_path"]
    image_dir = config_paths["image_path"]
    mask_dir = config_paths["mask_path"]

    # Removed the creation of top-level method directories

    pairs = recover_image_mask_pairs(root_dir=config_paths["root_path"])
    subset_size = config.get("subset", {}).get("subset_size", len(pairs))
    pairs = pairs[:subset_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    from unet import UNet
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    model.to(device)
    print("Model loaded.")

    overall_metrics = {}

    for idx, (img_path, mask_path) in enumerate(pairs):
        print(f"\nProcessing sample {idx}:")
        image = PIL.Image.open(img_path).convert("RGB")
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
        gt_mask = torchvision.transforms.ToTensor()(PIL.Image.open(mask_path)).cpu().numpy().squeeze()

        # Create a dedicated directory for this sample
        sample_root_dir = os.path.join(SAVE_PATH, f"sample_{idx}")
        os.makedirs(sample_root_dir, exist_ok=True)
        
        # Save original image and ground truth
        save_image(image_tensor, os.path.join(sample_root_dir, "original_image.png"))
        save_image(gt_mask, os.path.join(sample_root_dir, "ground_truth.png"))

        sample_metrics = {}

        # ----- Normal Inference -----
        normal_prob, normal_mask = normal_inference(model, image_tensor, activation=config_inference["activation"])
        normal_iou = compute_iou(normal_mask, gt_mask)
        normal_dice = compute_dice(normal_mask, gt_mask)
        normal_metrics = compute_metrics(normal_prob, gt_mask)
        normal_uncertainty = 1 - normal_prob
        normal_certainty = certainty_score(normal_uncertainty, gt_mask)
        sample_metrics["normal"] = {
            "iou": normal_iou,
            "dice": normal_dice,
            "certainty": normal_certainty,
            **normal_metrics
        }
        
        # Save normal inference results to sample's original subdir
        normal_dir = os.path.join(sample_root_dir, "original")
        os.makedirs(normal_dir, exist_ok=True)
        save_image(normal_prob, os.path.join(normal_dir, "probability.png"))
        save_image(normal_mask, os.path.join(normal_dir, "mask.png"))
        
        print(f"Normal Inference - IoU: {normal_iou:.4f}, Dice: {normal_dice:.4f}")

        # ----- MC Dropout Inference -----
        mc_dropout = MCDropout(model=model, p=config.get("mc_dropout", {}).get("p", 0.01))
        mc_dropout.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax, torch.nn.Sigmoid])
        mc_images, mc_masks, mc_mean_prediction, mc_entropy = mc_dropout_inference(
            model=model, image=image_tensor, num_samples=config_inference["num_samples"],
            activation=config_inference["activation"]
        )
        mc_dropout.remove()
        mc_mask_pred = (mc_mean_prediction > 0.5).astype(np.uint8)
        mc_iou = compute_iou(mc_mask_pred, gt_mask)
        mc_dice = compute_dice(mc_mask_pred, gt_mask)
        mc_metrics = compute_metrics(mc_mean_prediction, gt_mask)
        # After obtaining mc_entropy
        mc_certainty = certainty_score(mc_entropy, gt_mask)
        sample_metrics["mc_dropout"] = {
            "iou": mc_iou,
            "dice": mc_dice,
            "certainty": mc_certainty,
            **mc_metrics
        }
        
        # Save MC dropout results to sample's mc_dropout subdir
        mc_dropout_dir = os.path.join(sample_root_dir, "mc_dropout")
        save_outputs(mc_images, mc_masks, mc_mean_prediction, mc_entropy, mc_mask_pred, mc_dropout_dir, idx)
        
        print(f"MC Dropout - IoU: {mc_iou:.4f}, Dice: {mc_dice:.4f}")

        # ----- TTA Inference -----
        tta_images, tta_masks, tta_mean_prediction, tta_entropy = tta_inference(
            model=model, image=image_tensor, device=device, activation=config_inference["activation"]
        )
        tta_mask_pred = (tta_mean_prediction > 0.5).astype(np.uint8)
        tta_iou = compute_iou(tta_mask_pred, gt_mask)
        tta_dice = compute_dice(tta_mask_pred, gt_mask)
        tta_metrics = compute_metrics(tta_mean_prediction, gt_mask)
        # After obtaining tta_entropy
        tta_certainty = certainty_score(tta_entropy, gt_mask)
        sample_metrics["tta"] = {
            "iou": tta_iou,
            "dice": tta_dice,
            "certainty": tta_certainty,
            **tta_metrics
        }
        
        # Save TTA results to sample's tta subdir
        tta_dir = os.path.join(sample_root_dir, "tta")
        save_outputs(tta_images, tta_masks, tta_mean_prediction, tta_entropy, tta_mask_pred, tta_dir, idx)
        
        print(f"TTA Inference - IoU: {tta_iou:.4f}, Dice: {tta_dice:.4f}")

        # ----- Noisy Inference -----
        noisy_model = NoisyInference(
            image=image_tensor, N_SAMPLES=config_inference["num_samples"], noise_std=config_noisy["noise_std"]
        )
        noise_images, noise_masks, noise_mean_prediction, noise_entropy = noisy_inference(
            noisy_model=noisy_model, model=model, activation=config_inference["activation"]
        )
        noise_mask_pred = (noise_mean_prediction > 0.5).astype(np.uint8)
        noise_iou = compute_iou(noise_mask_pred, gt_mask)
        noise_dice = compute_dice(noise_mask_pred, gt_mask)
        noise_metrics = compute_metrics(noise_mean_prediction, gt_mask)
        # After obtaining noise_entropy
        noise_certainty = certainty_score(noise_entropy, gt_mask)
        sample_metrics["noisy"] = {
            "iou": noise_iou,
            "dice": noise_dice,
            "certainty": noise_certainty,
            **noise_metrics
        }
        
        # Save noisy inference results to sample's noisy subdir
        noisy_dir = os.path.join(sample_root_dir, "noisy")
        save_outputs(noise_images, noise_masks, noise_mean_prediction, noise_entropy, noise_mask_pred, noisy_dir, idx)
        
        print(f"Noisy Inference - IoU: {noise_iou:.4f}, Dice: {noise_dice:.4f}")

        # ----- Fusion of Uncertainty and Segmentation -----
        consensus_prob, consensus_uncertainty, consensus_mask = weighted_average_with_uncertainty(
            mc_mean_prediction, mc_entropy,
            tta_mean_prediction, tta_entropy,
            noise_mean_prediction, noise_entropy,
            weighting_method=config_fusion.get("weighting_method", "inverse"),
            beta=config_fusion.get("beta", 1.0),
            alpha=config_fusion.get("alpha", 1.0),
            threshold_method=config_fusion.get("threshold_method", "otsu"),
            percentile=config_fusion.get("percentile", 50),
            k=config_fusion.get("k", 0.5),
            epsilon=float(config_fusion.get("epsilon", 1e-6))
        )
        fusion_mask = (consensus_prob > 0.5).astype(np.uint8)
        fusion_iou = compute_iou(fusion_mask, gt_mask)
        fusion_dice = compute_dice(fusion_mask, gt_mask)
        fusion_metrics = compute_metrics(consensus_prob, gt_mask)
        # After obtaining consensus_uncertainty
        fusion_certainty = certainty_score(consensus_uncertainty, gt_mask)
        sample_metrics["fusion"] = {
            "iou": fusion_iou,
            "dice": fusion_dice,
            "certainty": fusion_certainty,
            **fusion_metrics
        }
        
        # Save fusion results to sample's fusion subdir
        fusion_dir = os.path.join(sample_root_dir, "fusion")
        os.makedirs(fusion_dir, exist_ok=True)
        save_image(consensus_prob, os.path.join(fusion_dir, "probability.png"))
        save_image(fusion_mask, os.path.join(fusion_dir, "mask.png"))
        save_image(consensus_uncertainty, os.path.join(fusion_dir, "uncertainty.png"))
        
        print(f"Fusion - IoU: {fusion_iou:.4f}, Dice: {fusion_dice:.4f}")

        # ----- CRF Refinement -----
        crf_probabilities, final_segmentation, final_uncertainty = refine_with_crf_uncertainty(
            image_tensor, consensus_prob, consensus_uncertainty,
            sdims=tuple(config_crf.get("sdims", (5, 5))),
            schan=tuple(config_crf.get("schan", (5, 5, 5))),
            n_iters=config_crf.get("n_iters", 5),
            epsilon=float(config_crf.get("epsilon", 1e-8))
        )
        if config_inference["activation"] == 'sigmoid':
            crf_mean_prob = crf_probabilities[1]
        else:
            crf_mean_prob = np.max(crf_probabilities, axis=0)
        crf_iou = compute_iou(final_segmentation, gt_mask)
        crf_dice = compute_dice(final_segmentation, gt_mask)
        crf_metrics = compute_metrics(crf_mean_prob, gt_mask)
        # After obtaining final_uncertainty
        crf_certainty = certainty_score(final_uncertainty, gt_mask)
        sample_metrics["crf"] = {
            "iou": crf_iou,
            "dice": crf_dice,
            "certainty": crf_certainty,
            **crf_metrics
        }
        
        # Save CRF results to sample's refined subdir
        crf_dir = os.path.join(sample_root_dir, "refined")
        os.makedirs(crf_dir, exist_ok=True)
        save_image(crf_mean_prob, os.path.join(crf_dir, "probability.png"))
        save_image(final_segmentation, os.path.join(crf_dir, "mask.png"))
        save_image(final_uncertainty, os.path.join(crf_dir, "uncertainty.png"))
        
        print(f"CRF Refinement - IoU: {crf_iou:.4f}, Dice: {crf_dice:.4f}")

        overall_metrics[f"sample_{idx}"] = sample_metrics

    # Create visualization directory
    visualization_dir = os.path.join(SAVE_PATH, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    # Aggregate metrics and save results
    methods = ['normal', 'mc_dropout', 'tta', 'noisy', 'fusion', 'crf']
    metrics = ['iou', 'dice', 'nll', 'ece', 'brier', 'accuracy', 'precision', 'recall', 'certainty']

    aggregated = {method: {metric: [] for metric in metrics} for method in methods}
    for sample_data in overall_metrics.values():
        for method in methods:
            if method in sample_data:
                for metric in metrics:
                    if metric in sample_data[method]:
                        aggregated[method][metric].append(sample_data[method][metric])

    # Compute mean metrics
    mean_results = {}
    for method in methods:
        mean_results[method] = {}
        for metric in metrics:
            if aggregated[method][metric]:
                mean_results[method][metric] = np.mean(aggregated[method][metric])
            else:
                mean_results[method][metric] = None

    # Save to CSV
    df = pd.DataFrame.from_dict(mean_results, orient='index')
    csv_path = os.path.join(visualization_dir, 'metrics_summary.csv')
    df.to_csv(csv_path)
    print(f"Metrics saved to {csv_path}")

    # Save detailed metrics for each sample
    sample_metrics_df = pd.DataFrame.from_dict({
        f"{sample_id}_{method}_{metric}": value
        for sample_id, methods_data in overall_metrics.items()
        for method, metrics_data in methods_data.items()
        for metric, value in metrics_data.items()
    }, orient='index')
    detailed_csv_path = os.path.join(visualization_dir, 'detailed_metrics.csv')
    sample_metrics_df.to_csv(detailed_csv_path, header=['value'])
    print(f"Detailed metrics saved to {detailed_csv_path}")

    # Plot metrics (initial plot)
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)
        plt.title(metric)
        values = [mean_results[method].get(metric, np.nan) for method in methods]
        plt.bar(methods, values)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(visualization_dir, 'metrics_summary.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Create enhanced subplots for better visualization of differences
    plt.figure(figsize=(18, 12))

    # Define colors for each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)
        values = [mean_results[method].get(metric, np.nan) for method in methods]
        
        # Use a bar plot with annotations
        bars = plt.bar(methods, values, color=colors)
        
        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Set consistent y-axis limits with increased top gap
        plt.ylim(0, 1.2 * max(values))  # Increase top gap by 20%
        plt.title(metric)
        plt.ylabel('Score')
        plt.xticks(rotation=45)

    # Add a single legend for all subplots
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=methods[i]) for i in range(len(methods))]
    plt.figlegend(handles=handles, labels=methods, loc='upper right', bbox_to_anchor=(1.1, 0.9), fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout()
    enhanced_plot_path = os.path.join(visualization_dir, 'enhanced_metrics_comparison.png')
    plt.savefig(enhanced_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Enhanced comparison plot saved to {enhanced_plot_path}")

    # Create box plots for each metric
    plt.figure(figsize=(18, 12))

    # Define colors for each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)
        
        # Prepare data for the box plot
        data = []
        for method in methods:
            # Extract all values for the current metric and method
            values = [sample_data[method].get(metric, np.nan) for sample_data in overall_metrics.values() if method in sample_data]
            data.append(values)
        
        # Create the box plot
        box = plt.boxplot(data, patch_artist=True, labels=methods)
        
        # Add colors to the boxes
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add title and labels
        plt.title(metric)
        plt.ylabel('Score')
        plt.xticks(rotation=45)

    # Add a legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=methods[i]) for i in range(len(methods))]
    plt.figlegend(handles=handles, labels=methods, loc='upper right', bbox_to_anchor=(1.1, 0.9), fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout()
    box_plot_path = os.path.join(visualization_dir, 'box_plot_comparison.png')
    plt.savefig(box_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Box plot comparison saved to {box_plot_path}")

    print("\nOverall Metrics:")
    for method in methods:
        print(f"Method: {method}")
        for metric in metrics:
            print(f"\t{metric}: {mean_results[method].get(metric, np.nan)}")