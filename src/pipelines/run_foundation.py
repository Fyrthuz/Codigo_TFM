"""Entry point for running UniVerSeg few-shot pipeline (patient-level split)."""
import argparse
import json
import os

import torch
import torchvision
from PIL import Image

from src.config import PipelineConfig
from src.pipelines.foundation import FoundationPipeline
from src.models.foundation import UniVerSegModel
from src.utils.dataset import recover_image_mask_pairs, load_test_indices


def load_g_channel_tensor(img_path, repeat=3):
    """Load image and extract the G channel (T1c - contrast-enhanced),
    then replicate it to form a 3-channel tensor.

    Rationale: UniVerSeg converts input to grayscale by averaging channels.
    Using only the G channel (T1c, where tumors enhance most) gives
    significantly better results than averaging all three modalities.
    """
    img = Image.open(img_path).convert("RGB")
    tensor = torchvision.transforms.ToTensor()(img)  # (3, H, W)
    g = tensor[1:2]  # keep dim: (1, H, W)
    return g.repeat(repeat, 1, 1)  # (3, H, W) with all channels = G


def main():
    parser = argparse.ArgumentParser(
        description="UniVerSeg few-shot segmentation pipeline"
    )
    parser.add_argument("--config", type=str, default="./configs/foundation_universeg.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--context-size", type=int, default=64,
                        help="Number of support examples for UniVerSeg few-shot")
    parser.add_argument("--test-indices", type=str, default="test_indices.json",
                        help="Path to test indices file")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    data_root = config.paths.get("root_path", "./MRI/filtered_data")
    pairs, patient_ids = recover_image_mask_pairs(root_dir=data_root)

    if os.path.exists(args.test_indices):
        test_idx = load_test_indices(args.test_indices)
        test_set = set(test_idx)
        train_pairs = [pairs[i] for i in range(len(pairs)) if i not in test_set]
        test_pairs = [pairs[i] for i in test_idx]
        print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
        context_pairs = train_pairs[:args.context_size]
        eval_pairs = test_pairs
    else:
        print(f"Warning: {args.test_indices} not found, using naive split")
        context_pairs = [(pairs[i], patient_ids[i]) for i in range(args.context_size)]
        eval_pairs = pairs[args.context_size:]

    # Build context tensors using G channel only (T1c)
    ctx_images, ctx_masks = [], []
    for item in context_pairs[:args.context_size]:
        if isinstance(item, tuple) and len(item) == 2:
            ip, mp = item
        else:
            ip, mp = item[0], item[1]
        ctx_images.append(load_g_channel_tensor(ip))
        ctx_masks.append(torchvision.transforms.ToTensor()(Image.open(mp)))
    ctx_images = torch.stack(ctx_images, dim=0)
    ctx_masks = torch.stack(ctx_masks, dim=0).float()

    model = UniVerSegModel(context_images=ctx_images, context_masks=ctx_masks)

    # Pass both support (context) and test pairs to pipeline
    # Test images also use G channel via the pipeline's load_sample
    support_pairs = [(p[0], p[1]) if isinstance(p, tuple) and len(p) == 2 else (p[0], p[1]) for p in context_pairs]
    dataset_info = {
        "pairs": eval_pairs,
        "support_pairs": support_pairs,
        "num_samples": len(eval_pairs),
    }
    pipeline = FoundationPipeline(config, model, dataset_info)
    pipeline.run()


if __name__ == "__main__":
    main()
