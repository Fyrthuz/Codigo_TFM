import os
from typing import Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm

from src.config import PipelineConfig
from src.pipelines.base import BaseSegmentationPipeline
from src.models.foundation import FoundationModel
from src.utils.metrics import compute_iou, compute_dice, compute_metrics, certainty_score
from src.utils.visualization import save_image


def load_g_channel_tensor(img_path, repeat=3):
    """Load image and extract the G channel (T1c - contrast-enhanced)."""
    from PIL import Image
    import torchvision
    img = Image.open(img_path).convert("RGB")
    tensor = torchvision.transforms.ToTensor()(img)
    g = tensor[1:2]
    return g.repeat(repeat, 1, 1)


class FoundationPipeline(BaseSegmentationPipeline):
    """Pipeline for UniVerSeg few-shot segmentation with full uncertainty support."""

    def __init__(
        self,
        config: PipelineConfig,
        foundation_model: FoundationModel,
        dataset_info: Dict,
    ):
        self.foundation_model = foundation_model
        self._pairs = dataset_info.get("pairs", [])
        self._support_pairs = dataset_info.get("support_pairs", [])
        self._dataset = dataset_info.get("dataset", None)
        self._n_samples = dataset_info.get("num_samples", 0)
        super().__init__(config)

    def build_model(self) -> torch.nn.Module:
        return self.foundation_model

    def get_num_samples(self) -> int:
        return self._n_samples

    def load_sample(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        if self._dataset is not None:
            image, mask = self._dataset[idx]
            if image.dim() == 4:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            mask = mask.numpy().squeeze()
            return image, mask

        img_path, mask_path = self._pairs[idx]
        from PIL import Image
        import torchvision
        image = Image.open(img_path).convert("RGB")
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        gt_mask = torchvision.transforms.ToTensor()(Image.open(mask_path)).cpu().numpy().squeeze()
        input_size = getattr(self.foundation_model, "input_size", None)
        if input_size is not None and gt_mask.shape != input_size:
            from PIL import Image as PILImage
            gt_mask_pil = PILImage.fromarray((gt_mask * 255).astype(np.uint8)).resize(input_size, PILImage.NEAREST)
            gt_mask = np.array(gt_mask_pil).astype(np.float32) / 255.0
        return image_tensor, gt_mask

    def run_normal_inference(self, image_tensor, gt_mask, activation="sigmoid"):
        with torch.no_grad():
            output = self.foundation_model(image_tensor)
        if activation == "sigmoid":
            prob = torch.sigmoid(output).squeeze(0).cpu().numpy()
            mask = (prob > 0.5).astype(np.uint8)
        elif activation == "softmax":
            prob = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
            mask = np.argmax(prob, axis=0)
        else:
            raise ValueError("activation must be 'sigmoid' or 'softmax'")

        iou = compute_iou(mask, gt_mask)
        dice = compute_dice(mask, gt_mask)
        metrics = compute_metrics(prob, gt_mask)
        uncertainty = 1 - prob
        cert = certainty_score(uncertainty, gt_mask)

        return {
            "prob": prob,
            "mask": mask,
            "uncertainty": uncertainty,
            "metrics": {"iou": iou, "dice": dice, "certainty": cert, **metrics},
        }

    def run_mc_dropout(self, image_tensor, gt_mask, config_inference, config_mc):
        try:
            return super().run_mc_dropout(image_tensor, gt_mask, config_inference, config_mc)
        except Exception as e:
            print(f"  MC Dropout skipped ({e})")
            return {"mean_prediction": np.zeros_like(gt_mask), "entropy": np.zeros_like(gt_mask),
                    "metrics": {"iou": 0.0, "dice": 0.0, "nll": 0.0, "ece": 0.0,
                                "brier": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "certainty": 0.0}}

    def run_tta(self, image_tensor, gt_mask, config_inference):
        try:
            return super().run_tta(image_tensor, gt_mask, config_inference)
        except Exception as e:
            print(f"  TTA skipped ({e})")
            return {"mean_prediction": np.zeros_like(gt_mask), "entropy": np.zeros_like(gt_mask),
                    "metrics": {"iou": 0.0, "dice": 0.0, "nll": 0.0, "ece": 0.0,
                                "brier": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "certainty": 0.0}}

    def run_noisy(self, image_tensor, gt_mask, config_inference, config_noisy):
        try:
            return super().run_noisy(image_tensor, gt_mask, config_inference, config_noisy)
        except Exception as e:
            print(f"  Noisy inference skipped ({e})")
            return {"mean_prediction": np.zeros_like(gt_mask), "entropy": np.zeros_like(gt_mask),
                    "metrics": {"iou": 0.0, "dice": 0.0, "nll": 0.0, "ece": 0.0,
                                "brier": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "certainty": 0.0}}

    def run_fusion(self, mc_result, tta_result, noisy_result, gt_mask, config_fusion):
        from src.utils.fusion import weighted_average_with_uncertainty

        valid_means, valid_uncerts = [], []
        for res in [mc_result, tta_result, noisy_result]:
            mp = res.get("mean_prediction")
            ent = res.get("entropy")
            if mp is not None and mp.ndim > 0 and res["metrics"]["dice"] > 0:
                valid_means.append(mp)
                valid_uncerts.append(ent)

        if len(valid_means) == 0:
            return {"prob": np.zeros(gt_mask.shape, dtype=np.float32),
                    "mask": np.zeros(gt_mask.shape, dtype=np.uint8),
                    "uncertainty": np.ones(gt_mask.shape, dtype=np.float32),
                    "metrics": {"iou": 0.0, "dice": 0.0, "nll": 0.0, "ece": 0.0,
                                "brier": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "certainty": 0.0}}

        while len(valid_means) < 3:
            valid_means.append(valid_means[-1])
            valid_uncerts.append(valid_uncerts[-1])

        mc_m, tta_m, noise_m = valid_means[:3]
        mc_u, tta_u, noise_u = valid_uncerts[:3]

        consensus_prob, consensus_uncertainty, consensus_mask = weighted_average_with_uncertainty(
            mc_m, mc_u, tta_m, tta_u, noise_m, noise_u,
            weighting_method=config_fusion.weighting_method,
            beta=config_fusion.beta, alpha=config_fusion.alpha,
            threshold_method=config_fusion.threshold_method,
            percentile=config_fusion.percentile, k=config_fusion.k,
            epsilon=config_fusion.epsilon,
        )
        fusion_mask = (consensus_prob > 0.5).astype(np.uint8)
        iou = compute_iou(fusion_mask, gt_mask)
        dice = compute_dice(fusion_mask, gt_mask)
        metrics_dict = compute_metrics(consensus_prob, gt_mask)
        cert = certainty_score(consensus_uncertainty, gt_mask)
        return {
            "prob": consensus_prob, "uncertainty": consensus_uncertainty,
            "mask": fusion_mask,
            "metrics": {"iou": iou, "dice": dice, "certainty": cert, **metrics_dict},
        }

    def _eval_pairs(self, pairs, label="test", save_path="./results"):
        """Evaluate a list of (img_path, mask_path) pairs."""
        n = len(pairs)
        overall = {}

        for idx in tqdm(range(n), desc=f"Processing {label} samples"):
            img_path, mask_path = pairs[idx]
            from PIL import Image
            import torchvision
            image_tensor = load_g_channel_tensor(img_path).unsqueeze(0).to(self.device)
            gt_mask = torchvision.transforms.ToTensor()(Image.open(mask_path)).cpu().numpy().squeeze()

            input_size = getattr(self.foundation_model, "input_size", None)
            if input_size is not None and gt_mask.shape != input_size:
                from PIL import Image as PILImage
                gt_mask_pil = PILImage.fromarray((gt_mask * 255).astype(np.uint8)).resize(input_size, PILImage.NEAREST)
                gt_mask = np.array(gt_mask_pil).astype(np.float32) / 255.0

            sample_dir = os.path.join(save_path, f"{label}_{idx}")
            os.makedirs(sample_dir, exist_ok=True)
            save_image(image_tensor, os.path.join(sample_dir, "original_image.png"))
            save_image(gt_mask, os.path.join(sample_dir, "ground_truth.png"))

            normal = self.run_normal_inference(image_tensor, gt_mask, self.config.inference.activation)
            self.save_sample_outputs(sample_dir, "original", normal)
            overall[f"{label}_{idx}"] = {"normal": normal["metrics"]}

            # Uncertainty methods
            mc = self.run_mc_dropout(image_tensor, gt_mask, self.config.inference, self.config.mc_dropout)
            overall[f"{label}_{idx}"]["mc_dropout"] = mc["metrics"]
            self.save_sample_outputs(sample_dir, "mc_dropout", mc)

            tta = self.run_tta(image_tensor, gt_mask, self.config.inference)
            overall[f"{label}_{idx}"]["tta"] = tta["metrics"]
            self.save_sample_outputs(sample_dir, "tta", tta)

            noisy = self.run_noisy(image_tensor, gt_mask, self.config.inference, self.config.noisy)
            overall[f"{label}_{idx}"]["noisy"] = noisy["metrics"]
            self.save_sample_outputs(sample_dir, "noisy", noisy)

            fusion = self.run_fusion(mc, tta, noisy, gt_mask, self.config.fusion)
            overall[f"{label}_{idx}"]["fusion"] = fusion["metrics"]
            self.save_sample_outputs(sample_dir, "fusion", fusion)

            crf = self.run_crf(image_tensor, fusion["prob"], fusion["uncertainty"], gt_mask, self.config.inference, self.config.crf)
            overall[f"{label}_{idx}"]["crf"] = crf["metrics"]
            self.save_sample_outputs(sample_dir, "refined", crf)

            dice_str = f"norm={normal['metrics']['dice']:.3f}"
            for name, res in [("mc", mc), ("tta", tta), ("noisy", noisy)]:
                if res["metrics"]["dice"] > 0:
                    dice_str += f" {name}={res['metrics']['dice']:.3f}"
            print(f"  {label} {idx} — Dice: {dice_str}")

        return overall

    def run(self):
        save_path = self.config.paths.get("save_path", "./results")

        model_name = type(self.foundation_model).__name__
        print(f"\n{'='*60}")
        print(f"Foundation Pipeline — Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        # Evaluate on SUPPORT set (same images as context)
        support_overall = {}
        if self._support_pairs:
            sp = self._support_pairs[:min(len(self._support_pairs), self.config.subset.subset_size or len(self._support_pairs))]
            print(f"\n--- SUPPORT SET ({len(sp)} images, seen during context) ---")
            support_overall = self._eval_pairs(sp, label="support", save_path=save_path)

        # Evaluate on TEST set (unseen patients)
        n_samples = self.get_num_samples()
        if self.config.subset.subset_size is not None:
            n_samples = min(n_samples, self.config.subset.subset_size)

        test_pairs = self._pairs[:n_samples]
        print(f"\n--- TEST SET ({len(test_pairs)} images, unseen patients) ---")
        test_overall = self._eval_pairs(test_pairs, label="test", save_path=save_path)

        # Consolidate and finalize
        self.overall_metrics = {**support_overall, **test_overall}
        self._finalize(save_path)

        # Print summary
        print(f"\n{'='*60}")
        print("SUPPORT SET (same images as context)")
        support_methods = {}
        for k, v in support_overall.items():
            for method, metrics in v.items():
                support_methods.setdefault(method, []).append(metrics["dice"])
        for method, dices in support_methods.items():
            print(f"  {method}: Dice = {np.mean(dices):.4f}")

        print("\nTEST SET (unseen patients)")
        test_methods = {}
        for k, v in test_overall.items():
            for method, metrics in v.items():
                test_methods.setdefault(method, []).append(metrics["dice"])
        for method, dices in test_methods.items():
            print(f"  {method}: Dice = {np.mean(dices):.4f}")

        return self.overall_metrics
