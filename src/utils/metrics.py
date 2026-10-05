import numpy as np


def compute_iou(pred, target, threshold: float = 0.5) -> float:
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    pred = (pred > threshold).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(pred, target, threshold: float = 0.5) -> float:
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    pred = (pred > threshold).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum + target_sum == 0:
        return 1.0
    return (2 * intersection) / (pred_sum + target_sum)


def classwise_ece(prob_flat, gt_flat, n_bins: int = 10) -> float:
    """Class-averaged Expected Calibration Error for binary segmentation.

    For each class c the class score s = P(y=c|x) is binned and the average
    score is compared against the empirical frequency of class c in that bin:

        ECE_c = sum_b (|B_b| / N) * |freq_c(B_b) - mean_score_c(B_b)|

    The final score averages ECE over both classes. Compared to the naive
    "bin accuracy (prediction == gt) vs mean probability" definition, this
    does not reward correctly predicted background pixels (which have
    near-zero foreground probability) with a ~1.0 error term, so it is not
    dominated by the huge background class and is meaningful for the
    strongly imbalanced foreground/background setting of lesion segmentation.
    """
    n = max(int(prob_flat.size), 1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_total = 0.0
    for cls in (0, 1):
        scores = prob_flat if cls == 1 else 1.0 - prob_flat
        correct = (gt_flat == cls).astype(np.float64)
        bin_indices = np.digitize(scores, bin_edges, right=True)
        ece_cls = 0.0
        for b in range(1, n_bins + 1):
            mask = bin_indices == b
            bin_size = int(np.sum(mask))
            if bin_size == 0:
                continue
            conf = float(np.mean(scores[mask]))
            freq = float(np.mean(correct[mask]))
            ece_cls += abs(freq - conf) * bin_size
        ece_total += ece_cls / n
    return ece_total / 2.0


def compute_metrics(prob, gt_mask, epsilon: float = 1e-8):
    # Remove leading singleton dimensions
    while prob.ndim > gt_mask.ndim and prob.shape[0] == 1:
        prob = prob[0]
    # Handle multi-class: convert to binary (tumor vs background)
    if prob.ndim > gt_mask.ndim and prob.shape[0] > 1:
        prob = prob.max(axis=0)
    prob_flat = prob.flatten()
    gt_flat = gt_mask.flatten().astype(np.int64)
    gt_flat = (gt_flat > 0).astype(np.int64)  # binarize for metrics
    prob_flat = np.clip(prob_flat, epsilon, 1 - epsilon)
    prob_flat = np.nan_to_num(prob_flat, nan=0.5)

    # Negative log-likelihood over all pixels...
    nll = -np.mean(
        gt_flat * np.log(prob_flat) + (1 - gt_flat) * np.log(1 - prob_flat)
    )
    # ...and restricted to foreground pixels, which is not dominated by
    # the (easy) background class and is the clinically relevant region.
    fg_pixels = gt_flat == 1
    if np.any(fg_pixels):
        nll_fg = float(-np.mean(np.log(prob_flat[fg_pixels])))
    else:
        nll_fg = float("nan")

    brier = np.mean((prob_flat - gt_flat) ** 2)

    pred_mask = (prob_flat > 0.5).astype(np.int64)

    tp = np.sum((pred_mask == 1) & (gt_flat == 1))
    fp = np.sum((pred_mask == 1) & (gt_flat == 0))
    tn = np.sum((pred_mask == 0) & (gt_flat == 0))
    fn = np.sum((pred_mask == 0) & (gt_flat == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    ece = classwise_ece(prob_flat, gt_flat)

    return {
        "nll": nll,
        "nll_fg": nll_fg,
        "ece": ece,
        "brier": brier,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def compute_ece(probs, targets, n_bins: int = 10) -> float:
    import torch
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
    confidences = probs_flat.max(dim=1)[0]
    predictions = probs_flat.argmax(dim=1)
    targets_flat = targets.flatten()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_indices = torch.bucketize(confidences, bin_boundaries, right=True)

    ece = 0.0
    for bin_idx in range(1, n_bins + 1):
        in_bin = bin_indices == bin_idx
        if in_bin.any():
            bin_acc = (predictions[in_bin] == targets_flat[in_bin]).float().mean()
            bin_conf = confidences[in_bin].mean()
            bin_weight = in_bin.float().mean()
            ece += torch.abs(bin_acc - bin_conf) * bin_weight

    return ece.item()


def certainty_score(uncertainty_map, ground_truth, num_classes: int = 2) -> float:
    uncertainty_map = np.asarray(uncertainty_map)
    ground_truth = np.squeeze(np.asarray(ground_truth))
    ground_truth = (ground_truth > 0.5).astype(bool)

    # Reduce multi-channel uncertainty to per-voxel (take mean across channels)
    if uncertainty_map.ndim > ground_truth.ndim:
        uncertainty_map = uncertainty_map.mean(axis=0)
    uncertainty_map = np.squeeze(uncertainty_map)

    if np.count_nonzero(ground_truth) == 0:
        return np.nan

    # Replace NaN/Inf in uncertainty map
    uncertainty_map = np.nan_to_num(uncertainty_map, nan=0.0, posinf=1.0, neginf=0.0)

    max_entropy = np.log(num_classes)
    if max_entropy == 0:
        return 1.0
    normalized_certainty = 1.0 - (uncertainty_map / max_entropy)
    normalized_certainty = np.clip(normalized_certainty, 0.0, 1.0)
    certainty_in_gt = normalized_certainty[ground_truth]
    return float(np.mean(certainty_in_gt))
