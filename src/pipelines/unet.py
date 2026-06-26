import json
import os
from typing import Tuple

import numpy as np
import torch
import torchvision
from PIL import Image

from src.config import PipelineConfig
from src.pipelines.base import BaseSegmentationPipeline
from src.models.unet import UNet
from src.utils.dataset import recover_image_mask_pairs, load_test_indices


class UNetPipeline(BaseSegmentationPipeline):
    def __init__(self, config: PipelineConfig, checkpoint_path: str = "unet_model.pth",
                 test_indices_path: str = "test_indices.json"):
        self.checkpoint_path = checkpoint_path
        all_pairs, all_pids = recover_image_mask_pairs(
            root_dir=config.paths.get("root_path", "./MRI/filtered_data")
        )
        # Filter by test indices if available
        if os.path.exists(test_indices_path):
            test_idx = load_test_indices(test_indices_path)
            self.pairs = [all_pairs[i] for i in test_idx]
            print(f"Using test set: {len(self.pairs)} samples (from {test_indices_path})")
        else:
            self.pairs = all_pairs
            print(f"Warning: {test_indices_path} not found, using all {len(self.pairs)} samples")
        super().__init__(config)

    def build_model(self) -> UNet:
        model = UNet(
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            init_features=self.config.model.init_features,
        )
        if os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state)
            print(f"Model loaded from {self.checkpoint_path}")
        else:
            print(f"Warning: checkpoint {self.checkpoint_path} not found, using untrained model")
        return model

    def get_num_samples(self) -> int:
        return len(self.pairs)

    def load_sample(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        gt_mask = torchvision.transforms.ToTensor()(Image.open(mask_path)).cpu().numpy().squeeze()
        return image_tensor, gt_mask

    @classmethod
    def from_yaml(cls, config_path: str, checkpoint_path: str = "unet_model.pth",
                  test_indices_path: str = "test_indices.json") -> "UNetPipeline":
        config = PipelineConfig.from_yaml(config_path)
        return cls(config, checkpoint_path=checkpoint_path, test_indices_path=test_indices_path)
