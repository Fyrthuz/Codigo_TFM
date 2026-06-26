import numpy as np
from skimage.filters import threshold_otsu


def dynamic_threshold_multiclass(prob_maps, method="otsu", percentile=50, k=0.5, epsilon=1e-6):
    if prob_maps.ndim == 2:
        if method.lower() == "otsu":
            threshold = threshold_otsu(prob_maps)
        elif method.lower() == "percentile":
            threshold = np.percentile(prob_maps, percentile)
        elif method.lower() == "mean_std":
            threshold = np.mean(prob_maps) + k * np.std(prob_maps)
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
                thresholds[c] = np.mean(class_probs) + k * np.std(class_probs)
            else:
                raise ValueError("Unknown method. Choose from 'otsu', 'percentile', or 'mean_std'.")
        return thresholds
    else:
        raise ValueError("prob_maps must be either a 2D (binary) or 3D (multiclass) array.")


def weighted_average_with_uncertainty(
    mc_mean, mc_uncert, tta_mean, tta_uncert, noise_mean, noise_uncert,
    weighting_method="inverse", beta=1.0, alpha=1.0,
    threshold_method="otsu", percentile=50, k=0.5, epsilon=1e-6,
):
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
    else:
        num_classes = consensus_prob.shape[0]

    if threshold_method == "naive":
        if num_classes == 1:
            consensus_mask = (consensus_prob > 0.5).astype(np.uint8)
        else:
            consensus_mask = np.argmax(consensus_prob, axis=0)
    else:
        thresholds = dynamic_threshold_multiclass(
            consensus_prob, method=threshold_method, percentile=percentile, k=k
        )
        if len(thresholds) == 1:
            consensus_mask = (consensus_prob > thresholds[0]).astype(np.uint8)
        else:
            class_masks = np.zeros((num_classes, *consensus_prob.shape[1:]), dtype=np.uint8)
            for c in range(num_classes):
                class_masks[c] = (consensus_prob[c] > thresholds[c]).astype(np.uint8)
            consensus_mask = np.argmax(class_masks, axis=0)

    return consensus_prob, consensus_uncertainty, consensus_mask
