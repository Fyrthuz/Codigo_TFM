import numpy as np


def refine_with_crf_uncertainty(
    image, prob_map, uncertainty_map,
    sdims=(5, 5), schan=(5, 5, 5), n_iters=5, epsilon=1e-8,
):
    """Dense CRF refinement using numpy/OpenCV (no pydensecrf needed).

    Implements Krähenbühl & Koltun 2012 mean-field inference:
      1. Gaussian kernel (smoothness)
      2. Bilateral kernel (appearance)
    """
    # Convert image to numpy uint8
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    if image.ndim == 5:
        image = image[0]
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 4:
        image = np.transpose(image, (1, 2, 3, 0))
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[..., :3]
    image = image.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Prepare probability stack
    if prob_map.ndim == 2:
        prob_stack = np.stack([1 - prob_map, prob_map], axis=0)
        n_classes = 2
    elif prob_map.ndim == 3:
        prob_stack = prob_map
        n_classes = prob_map.shape[0]
    else:
        raise ValueError("prob_map must be 2D (binary) or 3D (multiclass)")

    H, W = prob_stack.shape[1:]

    # Add epsilon and normalize
    Q = prob_stack.copy().astype(np.float64)
    Q = np.clip(Q, epsilon, 1 - epsilon)
    Q = Q / (Q.sum(axis=0, keepdims=True) + epsilon)

    # Uncertainty-weighted unary adjustment
    norm_uncert = (uncertainty_map - np.min(uncertainty_map)) / (np.ptp(uncertainty_map) + epsilon)
    uniform_unary = np.ones((n_classes, H, W), dtype=np.float64) / n_classes
    adjusted_prob = (1 - norm_uncert) * Q + norm_uncert * uniform_unary
    adjusted_prob = np.clip(adjusted_prob, epsilon, 1 - epsilon)
    adjusted_prob = adjusted_prob / (adjusted_prob.sum(axis=0, keepdims=True) + epsilon)

    Q = adjusted_prob.copy()

    # CRF mean-field iterations
    try:
        import cv2
        use_cv2 = True
    except ImportError:
        use_cv2 = False

    for _ in range(n_iters):
        Q_tilde = np.zeros_like(Q)

        for c in range(n_classes):
            q_c = Q[c]

            # 1. Gaussian smoothness kernel
            if use_cv2:
                smooth = cv2.GaussianBlur(q_c, (0, 0), sigmaX=sdims[0])
            else:
                from scipy.ndimage import gaussian_filter
                smooth = gaussian_filter(q_c, sigma=sdims[0])

            # 2. Bilateral appearance kernel
            if use_cv2:
                bilateral = cv2.bilateralFilter(
                    q_c.astype(np.float32), d=-1,
                    sigmaColor=schan[0],
                    sigmaSpace=sdims[1],
                )
            else:
                bilateral = smooth  # fallback

            # Combine kernels (Potts model, weights 5 each)
            Q_tilde[c] = smooth * 5 + bilateral * 5

        # Apply message in log-space: Q *= exp(Q_tilde)
        Q = adjusted_prob * np.exp(Q_tilde)
        Q = np.clip(Q, epsilon, 1 - epsilon)
        Q = Q / (Q.sum(axis=0, keepdims=True) + epsilon)

    # Output
    refined_segmentation = np.argmax(Q, axis=0)
    refined_uncertainty = -np.sum(Q * np.log(Q + epsilon), axis=0)

    return Q, refined_segmentation, refined_uncertainty
