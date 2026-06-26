import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

from .base import FoundationModel


class UniVerSegModel(FoundationModel):
    """UniVerSeg few-shot segmentation model.

    Requires a support (context) set of image-mask pairs at inference.
    Works on 2D images; converts inputs to grayscale and resizes to 128x128.

    Installation:
        pip install git+https://github.com/JJGO/UniverSeg.git
    """

    def __init__(
        self,
        context_images: Optional[torch.Tensor] = None,
        context_masks: Optional[torch.Tensor] = None,
        device: torch.device = None,
        input_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        try:
            from universeg import universeg
        except ImportError:
            raise ImportError(
                "UniVerSeg is required. Install with:\n"
                "  pip install git+https://github.com/JJGO/UniverSeg.git"
            )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model = universeg(pretrained=True).to(self.device).eval()
        self._context_images = None
        self._context_masks = None
        if context_images is not None and context_masks is not None:
            self.set_context(context_images, context_masks)

    def set_context(self, images: torch.Tensor, masks: torch.Tensor):
        """Set the support set for few-shot inference.

        Args:
            images: (N, 3, H, W) tensor of support images (batch of RGB).
            masks:  (N, 1, H, W) tensor of support masks.
        """
        images = self._preprocess(images, to_grayscale=True)
        masks = masks.to(self.device).float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        # Resize masks to input_size
        if masks.shape[-2:] != self.input_size:
            masks = F.interpolate(masks, size=self.input_size, mode="bilinear", align_corners=False)
        # UniVerSeg expects (B, N, 1, H, W)
        images = images.unsqueeze(0)  # (N, 1, H, W) → (1, N, 1, H, W)
        masks = masks.unsqueeze(0)    # (N, 1, H, W) → (1, N, 1, H, W)
        self._context_images = images
        self._context_masks = masks

    def _preprocess(self, x: torch.Tensor, to_grayscale: bool = True) -> torch.Tensor:
        """Convert to grayscale and resize to input_size. Expects (B, C, H, W) or (B, H, W)."""
        x = x.to(self.device).float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if to_grayscale and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self._context_images is None:
            raise RuntimeError(
                "No context set. Call set_context(images, masks) with support examples first."
            )
        image = self._preprocess(image)
        # Ensure (B, 1, H, W)
        if image.shape[1] != 1:
            image = image.mean(dim=1, keepdim=True)
        with torch.no_grad():
            logits = self.model(image, self._context_images, self._context_masks)
        return logits

    def get_output_channels(self) -> int:
        return 1 if self._context_masks is None else self._context_masks.shape[1]
