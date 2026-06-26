import math
import torch
import torch.nn.functional as F
import numpy as np


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


def tta_inference(model, image, device: str, activation: str = "sigmoid"):
    import ttach as tta_lib
    import torch.nn.functional as F
    import numpy as np

    transforms = tta_lib.Compose([
        tta_lib.HorizontalFlip(),
        tta_lib.Scale(scales=[0.5, 1, 2]),
        tta_lib.Multiply(factors=[0.8, 0.9, 1, 1.1, 1.2]),
    ])

    tta_predictions = []
    augmented_images = []

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
