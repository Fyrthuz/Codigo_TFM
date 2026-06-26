import torch
import torch.nn as nn


class MCDropout:
    def __init__(self, model: nn.Module, p: float = 0.2):
        self.model = model
        self.p = p
        self.hooks = []
        self.enabled = False

    def _apply_mask(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            mask = (torch.rand_like(output) > self.p).float()
            return output * mask
        return output

    def enable(
        self,
        ignore_specific_layers: list = None,
        ignore_type_layers: list = None,
        layer_types: list = None,
    ):
        if self.enabled:
            return

        ignore_specific_layers = ignore_specific_layers or []
        ignore_type_layers = ignore_type_layers or []
        layer_types = layer_types or []

        for name, module in self.model.named_modules():
            if name == "":
                continue
            apply_condition = (
                (isinstance(module, tuple(layer_types)) or not layer_types)
                and module not in ignore_specific_layers
                and not isinstance(module, tuple(ignore_type_layers))
            )
            if apply_condition:
                self.hooks.append(module.register_forward_hook(self._apply_mask))

        self.enabled = True

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.enabled = False


def mc_dropout_inference(model, image, num_samples, activation="sigmoid"):
    import torch.nn.functional as F
    import numpy as np

    images = []
    masks_list = []

    with torch.no_grad():
        outputs = [model(image) for _ in range(num_samples)]
        images.extend([image.cpu().numpy() for _ in range(num_samples)])

    outputs = torch.stack(outputs)

    if activation == "softmax":
        # outputs shape: (num_samples, B, C, ...). Softmax along class dim (2).
        softmax_preds = F.softmax(outputs, dim=2)
        mean_probs = softmax_preds.mean(dim=0).cpu().numpy()
        entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)
        masks_list = np.argmax(softmax_preds.cpu().numpy(), axis=2)
    elif activation == "sigmoid":
        sigmoid_preds = torch.sigmoid(outputs)
        mean_probs = np.squeeze(sigmoid_preds.mean(dim=0).cpu().numpy())
        entropy_map = -(
            mean_probs * np.log(mean_probs + 1e-8)
            + (1 - mean_probs) * np.log(1 - mean_probs + 1e-8)
        )
        entropy_map = np.squeeze(entropy_map)
        masks_list = (sigmoid_preds.cpu().numpy() > 0.5).astype(np.uint8)
    else:
        raise ValueError("activation must be 'softmax' or 'sigmoid'")

    return images, masks_list, mean_probs, entropy_map
