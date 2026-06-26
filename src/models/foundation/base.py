from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class NoTrainingRequired(ABC):
    """Marker mixin: models that work zero-shot / few-shot without training."""


class FoundationModel(nn.Module, NoTrainingRequired, ABC):
    """Abstract base for all foundation segmentation models.

    These models require NO training — they work either zero-shot (pretrained
    on large datasets) or few-shot (using a support/context set at inference).

    Subclasses must implement:
        forward(image) -> torch.Tensor  (logits or probabilities)
    """

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_output_channels(self) -> int:
        ...

    def predict(self, image: torch.Tensor, activation: str = "sigmoid") -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (probabilities, hard_mask)."""
        logits = self.forward(image)
        if activation == "sigmoid":
            prob = torch.sigmoid(logits)
            mask = (prob > 0.5).float()
        elif activation == "softmax":
            prob = torch.softmax(logits, dim=1)
            mask = prob.argmax(dim=1, keepdim=True).float()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        return prob, mask
