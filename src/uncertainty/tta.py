import math

import numpy as np
import torch
import torch.nn.functional as F


class RandomImageTransformer:
    def __init__(
        self,
        degrees=(-30, 30),
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=(-10, 10),
        padding_mode="border",
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.padding_mode = padding_mode

    def _get_forward_affine_matrix(self, center, angle, translate, scale, shear):
        cx, cy = center
        tx, ty = translate
        angle_rad = math.radians(angle)
        shear_rad = math.radians(shear)

        R = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad),  math.cos(angle_rad), 0],
            [0, 0, 1],
        ])

        S = torch.tensor([
            [1, math.tan(shear_rad), 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        Sc = torch.diag(torch.tensor([scale, scale, 1.0]))

        T_center = torch.tensor([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1],
        ])
        T_neg_center = torch.tensor([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1],
        ])
        T_translation = torch.tensor([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ])

        M = T_translation @ T_center @ R @ S @ Sc @ T_neg_center
        return M

    def _get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        M_fwd = self._get_forward_affine_matrix(center, angle, translate, scale, shear)
        M_inv = torch.inverse(M_fwd)
        return M_inv

    def transform_image(self, image_tensor, return_matrix=False):
        _, H, W = image_tensor.shape
        angle = random.uniform(*self.degrees)
        tx = random.uniform(-self.translate[0] * W, self.translate[0] * W)
        ty = random.uniform(-self.translate[1] * H, self.translate[1] * H)
        s = random.uniform(*self.scale)
        sh = random.uniform(*self.shear)

        cx, cy = W / 2.0, H / 2.0

        M_inv = self._get_inverse_affine_matrix((cx, cy), angle, (tx, ty), s, sh)
        M_inv_2x3 = M_inv[:2, :]

        grid = F.affine_grid(
            M_inv_2x3.unsqueeze(0),
            image_tensor.unsqueeze(0).shape,
            align_corners=False,
        )
        transformed = F.grid_sample(
            image_tensor.unsqueeze(0),
            grid,
            padding_mode=self.padding_mode,
            align_corners=False,
        )

        transformed = transformed.squeeze(0)
        if return_matrix:
            return transformed, (M_fwd := self._get_forward_affine_matrix((cx, cy), angle, (tx, ty), s, sh))
        return transformed

    def restore_image(self, image_tensor, M_inv):
        _, H, W = image_tensor.shape
        M_inv_2x3 = M_inv[:2, :]
        grid = F.affine_grid(
            M_inv_2x3.unsqueeze(0),
            image_tensor.unsqueeze(0).shape,
            align_corners=False,
        )
        restored = F.grid_sample(
            image_tensor.unsqueeze(0),
            grid,
            padding_mode="zeros",
            align_corners=False,
        )
        return restored.squeeze(0)

    def __call__(self, image_tensor):
        return self.transform_image(image_tensor, return_matrix=False)


import random


def _resize_safe_combinations():
    """Size- and orientation-preserving test-time transforms.

    Two constraints apply to models with a fixed internal input size
    (e.g. UniVerSeg resizes everything to 128x128):

    1. Size-preserving: ttach's Scale transforms break there, because
       deaugmentation assumes the model output has the same size as the
       augmented input; with internal resizing the deaugmented masks come
       back as a mixture of sizes (256/128/64 for 256x256 inputs) and
       ``torch.stack`` fails — the original "TTA no disponible" issue.
    2. Orientation-preserving: for in-context models with a fixed support
       set, flipping only the query breaks query-support matching
       (measured on LGG: flip-averaged TTA drops support Dice from 0.94
       to 0.14). Flipping query AND support together reproduces the
       original prediction exactly (Dice 0.97 against itself) and adds no
       diversity, so only photometric transforms yield valid, diverse
       predictions for such models.
    """
    def _mul(x, f):
        return x * f

    def _gamma(x, g):
        return torch.clamp(x, 1e-6, 1.0) ** g

    def _bias(x, b):
        return x + b

    def _contrast(x, c):
        return (x - 0.5) * c + 0.5

    return [
        ("identity", lambda x: x),
        ("mul_x0.9", lambda x: _mul(x, 0.9)),
        ("mul_x1.1", lambda x: _mul(x, 1.1)),
        ("gamma_0.85", lambda x: _gamma(x, 0.85)),
        ("gamma_1.15", lambda x: _gamma(x, 1.15)),
        ("bias_-0.03", lambda x: _bias(x, -0.03)),
        ("bias_+0.03", lambda x: _bias(x, 0.03)),
        ("contrast_x0.9", lambda x: _contrast(x, 0.9)),
        ("contrast_x1.1", lambda x: _contrast(x, 1.1)),
    ]


_RESIZE_SAFE_COMBOS = _resize_safe_combinations()


def tta_inference(model, image, device: str, activation: str = "sigmoid", resize_safe: bool = False):
    """Test-Time Augmentation inference.

    Args:
        model: segmentation model. Must accept (1, C, H, W) tensors.
        image: (1, C, H, W) input tensor.
        device: device string (kept for API compatibility).
        activation: "sigmoid" (binary) or "softmax" (multi-class).
        resize_safe: use only size- and orientation-preserving photometric
            transforms (required for models with a fixed internal input
            size and/or a fixed in-context support set, where ttach's
            Scale/Flip transforms are incompatible — see
            ``_resize_safe_combinations``).
    """
    import torch.nn.functional as F

    tta_predictions = []
    augmented_images = []

    if resize_safe:
        with torch.no_grad():
            for _name, transform in _RESIZE_SAFE_COMBOS:
                augmented_image = torch.clamp(transform(image), 0.0, 1.0)
                augmented_images.append(augmented_image.cpu().numpy())
                output = model(augmented_image)
                tta_predictions.append(output)
    else:
        import ttach as tta_lib

        transforms = tta_lib.Compose([
            tta_lib.HorizontalFlip(),
            tta_lib.Scale(scales=[0.5, 1, 2]),
            tta_lib.Multiply(factors=[0.8, 0.9, 1, 1.1, 1.2]),
        ])

        with torch.no_grad():
            for transform in transforms:
                augmented_image = transform.augment_image(image)
                augmented_images.append(augmented_image.cpu().numpy())
                output = model(augmented_image)
                output = transform.deaugment_mask(output)
                tta_predictions.append(output)

    tta_predictions = torch.stack(tta_predictions)

    if activation == "softmax":
        softmax_preds = F.softmax(tta_predictions, dim=2)
        mean_probs = softmax_preds.mean(dim=0).cpu().numpy()
        entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)
        masks_list = np.argmax(softmax_preds.cpu().numpy(), axis=2)
    elif activation == "sigmoid":
        sigmoid_preds = torch.sigmoid(tta_predictions)
        mean_probs = sigmoid_preds.mean(dim=0).cpu().numpy()
        entropy_map = -(
            mean_probs * np.log(mean_probs + 1e-8)
            + (1 - mean_probs) * np.log(1 - mean_probs + 1e-8)
        )
        masks_list = (sigmoid_preds.cpu().numpy() > 0.5).astype(np.uint8)
    else:
        raise ValueError("activation must be 'softmax' or 'sigmoid'")

    return augmented_images, masks_list, mean_probs, entropy_map
