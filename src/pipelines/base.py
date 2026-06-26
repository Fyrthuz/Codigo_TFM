import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import PipelineConfig
from src.utils.metrics import compute_iou, compute_dice, compute_metrics, certainty_score
from src.utils.fusion import weighted_average_with_uncertainty
from src.utils.crf import refine_with_crf_uncertainty
from src.utils.visualization import (
    save_image,
    plot_metrics_comparison,
    plot_enhanced_comparison,
    plot_box_comparison,
    save_metrics_csv,
)
from src.uncertainty.mc_dropout import MCDropout, mc_dropout_inference
from src.uncertainty.tta import tta_inference
from src.uncertainty.noise_inference import NoisyInference, noisy_inference


class BaseSegmentationPipeline(ABC):
    METHODS = ["normal", "mc_dropout", "tta", "noisy", "fusion", "crf"]
    METRICS = ["iou", "dice", "nll", "ece", "brier", "accuracy", "precision", "recall", "certainty"]

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.overall_metrics: Dict[str, Dict] = {}

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        ...

    @abstractmethod
    def load_sample(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        ...

    @abstractmethod
    def get_num_samples(self) -> int:
        ...

    def run_normal_inference(self, image_tensor: torch.Tensor, gt_mask: np.ndarray, activation: str = "sigmoid"):
        with torch.no_grad():
            output = self.model(image_tensor)
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

    def run_mc_dropout(self, image_tensor: torch.Tensor, gt_mask: np.ndarray, config_inference, config_mc):
        p = getattr(config_mc, "p", 0.01)
        mc = MCDropout(model=self.model, p=p)
        mc.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax, torch.nn.Sigmoid])
        mc_images, mc_masks, mc_mean_prediction, mc_entropy = mc_dropout_inference(
            model=self.model, image=image_tensor,
            num_samples=config_inference.num_samples,
            activation=config_inference.activation,
        )
        mc.remove()
        mc_mask_pred = (mc_mean_prediction > 0.5).astype(np.uint8)
        iou = compute_iou(mc_mask_pred, gt_mask)
        dice = compute_dice(mc_mask_pred, gt_mask)
        metrics = compute_metrics(mc_mean_prediction, gt_mask)
        cert = certainty_score(mc_entropy, gt_mask)
        return {
            "images": mc_images,
            "masks": mc_masks,
            "mean_prediction": mc_mean_prediction,
            "entropy": mc_entropy,
            "mask_pred": mc_mask_pred,
            "metrics": {"iou": iou, "dice": dice, "certainty": cert, **metrics},
        }

    def run_tta(self, image_tensor: torch.Tensor, gt_mask: np.ndarray, config_inference):
        tta_images, tta_masks, tta_mean_prediction, tta_entropy = tta_inference(
            model=self.model, image=image_tensor, device=str(self.device),
            activation=config_inference.activation,
        )
        tta_mask_pred = (tta_mean_prediction > 0.5).astype(np.uint8)
        iou = compute_iou(tta_mask_pred, gt_mask)
        dice = compute_dice(tta_mask_pred, gt_mask)
        metrics = compute_metrics(tta_mean_prediction, gt_mask)
        cert = certainty_score(tta_entropy, gt_mask)
        return {
            "images": tta_images,
            "masks": tta_masks,
            "mean_prediction": tta_mean_prediction,
            "entropy": tta_entropy,
            "mask_pred": tta_mask_pred,
            "metrics": {"iou": iou, "dice": dice, "certainty": cert, **metrics},
        }

    def run_noisy(self, image_tensor: torch.Tensor, gt_mask: np.ndarray, config_inference, config_noisy):
        noisy_model = NoisyInference(
            image=image_tensor,
            N_SAMPLES=config_inference.num_samples,
            noise_std=config_noisy.noise_std,
        )
        noise_images, noise_masks, noise_mean_prediction, noise_entropy = noisy_inference(
            noisy_model=noisy_model, model=self.model,
            activation=config_inference.activation,
        )
        noise_mask_pred = (noise_mean_prediction > 0.5).astype(np.uint8)
        iou = compute_iou(noise_mask_pred, gt_mask)
        dice = compute_dice(noise_mask_pred, gt_mask)
        metrics = compute_metrics(noise_mean_prediction, gt_mask)
        cert = certainty_score(noise_entropy, gt_mask)
        return {
            "images": noise_images,
            "masks": noise_masks,
            "mean_prediction": noise_mean_prediction,
            "entropy": noise_entropy,
            "mask_pred": noise_mask_pred,
            "metrics": {"iou": iou, "dice": dice, "certainty": cert, **metrics},
        }

    def run_fusion(self, mc_result, tta_result, noisy_result, gt_mask, config_fusion):
        consensus_prob, consensus_uncertainty, consensus_mask = weighted_average_with_uncertainty(
            mc_result["mean_prediction"], mc_result["entropy"],
            tta_result["mean_prediction"], tta_result["entropy"],
            noisy_result["mean_prediction"], noisy_result["entropy"],
            weighting_method=config_fusion.weighting_method,
            beta=config_fusion.beta,
            alpha=config_fusion.alpha,
            threshold_method=config_fusion.threshold_method,
            percentile=config_fusion.percentile,
            k=config_fusion.k,
            epsilon=config_fusion.epsilon,
        )
        fusion_mask = (consensus_prob > 0.5).astype(np.uint8)
        iou = compute_iou(fusion_mask, gt_mask)
        dice = compute_dice(fusion_mask, gt_mask)
        metrics = compute_metrics(consensus_prob, gt_mask)
        cert = certainty_score(consensus_uncertainty, gt_mask)
        return {
            "prob": consensus_prob,
            "uncertainty": consensus_uncertainty,
            "mask": fusion_mask,
            "metrics": {"iou": iou, "dice": dice, "certainty": cert, **metrics},
        }

    def run_crf(self, image_tensor, consensus_prob, consensus_uncertainty, gt_mask, config_inference, config_crf):
        try:
            crf_probabilities, final_segmentation, final_uncertainty = refine_with_crf_uncertainty(
                image_tensor, consensus_prob, consensus_uncertainty,
                sdims=config_crf.sdims,
                schan=config_crf.schan,
                n_iters=config_crf.n_iters,
                epsilon=config_crf.epsilon,
            )
        except ImportError:
            print("  pydensecrf not installed — skipping CRF refinement")
            return {"prob": consensus_prob, "segmentation": (consensus_prob > 0.5).astype(np.uint8),
                    "uncertainty": consensus_uncertainty, "metrics": {"iou": 0.0, "dice": 0.0, "nll": 0.0,
                    "ece": 0.0, "brier": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "certainty": 0.0}}
        if config_inference.activation == "sigmoid":
            crf_mean_prob = crf_probabilities[1]
        else:
            crf_mean_prob = np.max(crf_probabilities, axis=0)
        iou = compute_iou(final_segmentation, gt_mask)
        dice = compute_dice(final_segmentation, gt_mask)
        metrics = compute_metrics(crf_mean_prob, gt_mask)
        cert = certainty_score(final_uncertainty, gt_mask)
        return {
            "prob": crf_mean_prob,
            "segmentation": final_segmentation,
            "uncertainty": final_uncertainty,
            "metrics": {"iou": iou, "dice": dice, "certainty": cert, **metrics},
        }

    def save_sample_outputs(self, sample_dir: str, name: str, data: dict):
        method_dir = os.path.join(sample_dir, name)
        os.makedirs(method_dir, exist_ok=True)

        if "prob" in data:
            save_image(data["prob"], os.path.join(method_dir, "probability.png"))
        if "mask" in data:
            save_image(data["mask"], os.path.join(method_dir, "mask.png"))
        if "uncertainty" in data:
            save_image(data["uncertainty"], os.path.join(method_dir, "uncertainty.png"))

        if name in ("mc_dropout", "tta", "noisy"):
            images_dir = os.path.join(method_dir, "images")
            predictions_dir = os.path.join(method_dir, "predictions")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(predictions_dir, exist_ok=True)

            if "images" in data and isinstance(data["images"], list):
                for i, img in enumerate(data["images"]):
                    save_image(img, os.path.join(images_dir, f"image_{i}.png"))
            if "masks" in data and isinstance(data["masks"], (list, np.ndarray)):
                masks_list = [np.asarray(m) for m in data["masks"]] if isinstance(data["masks"], list) else data["masks"]
                for i, m in enumerate(masks_list):
                    save_image(m, os.path.join(predictions_dir, f"prediction_{i}.png"))

            if "mean_prediction" in data:
                save_image(data["mean_prediction"], os.path.join(method_dir, "mean_prediction.png"))
            if "entropy" in data:
                save_image(data["entropy"], os.path.join(method_dir, "uncertainty.png"))
            if "mask_pred" in data:
                save_image(data["mask_pred"], os.path.join(method_dir, "mean_mask_prediction.png"))

    def run(self):
        save_path = self.config.paths.get("save_path", "./results")
        subset_size = self.config.subset.subset_size
        n_samples = self.get_num_samples()
        if subset_size is not None:
            n_samples = min(n_samples, subset_size)

        for idx in tqdm(range(n_samples), desc="Processing samples"):
            image_tensor, gt_mask = self.load_sample(idx)
            sample_dir = os.path.join(save_path, f"sample_{idx}")
            os.makedirs(sample_dir, exist_ok=True)

            save_image(image_tensor, os.path.join(sample_dir, "original_image.png"))
            save_image(gt_mask, os.path.join(sample_dir, "ground_truth.png"))

            sample_metrics = {}
            cfg_inf = self.config.inference

            normal = self.run_normal_inference(image_tensor, gt_mask, cfg_inf.activation)
            sample_metrics["normal"] = normal["metrics"]
            self.save_sample_outputs(sample_dir, "original", normal)

            mc = self.run_mc_dropout(image_tensor, gt_mask, cfg_inf, self.config.mc_dropout)
            sample_metrics["mc_dropout"] = mc["metrics"]
            self.save_sample_outputs(sample_dir, "mc_dropout", mc)

            tta = self.run_tta(image_tensor, gt_mask, cfg_inf)
            sample_metrics["tta"] = tta["metrics"]
            self.save_sample_outputs(sample_dir, "tta", tta)

            noisy = self.run_noisy(image_tensor, gt_mask, cfg_inf, self.config.noisy)
            sample_metrics["noisy"] = noisy["metrics"]
            self.save_sample_outputs(sample_dir, "noisy", noisy)

            fusion = self.run_fusion(mc, tta, noisy, gt_mask, self.config.fusion)
            sample_metrics["fusion"] = fusion["metrics"]
            self.save_sample_outputs(sample_dir, "fusion", fusion)

            crf = self.run_crf(image_tensor, fusion["prob"], fusion["uncertainty"], gt_mask, cfg_inf, self.config.crf)
            sample_metrics["crf"] = crf["metrics"]
            self.save_sample_outputs(sample_dir, "refined", crf)

            self.overall_metrics[f"sample_{idx}"] = sample_metrics

        self._finalize(save_path)
        return self.overall_metrics

    def _finalize(self, save_path: str):
        viz_dir = os.path.join(save_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Only include methods that actually have data
        active_methods = []
        for method in self.METHODS:
            for sample_data in self.overall_metrics.values():
                if method in sample_data:
                    active_methods.append(method)
                    break

        active_metrics = []
        for metric in self.METRICS:
            for sample_data in self.overall_metrics.values():
                for method in active_methods:
                    if metric in sample_data.get(method, {}):
                        active_metrics.append(metric)
                        break
                if metric in active_metrics:
                    break

        aggregated = {
            method: {metric: [] for metric in active_metrics}
            for method in active_methods
        }
        for sample_data in self.overall_metrics.values():
            for method in active_methods:
                if method in sample_data:
                    for metric in active_metrics:
                        if metric in sample_data[method]:
                            aggregated[method][metric].append(sample_data[method][metric])

        mean_results = {}
        for method in active_methods:
            mean_results[method] = {}
            for metric in active_metrics:
                vals = aggregated[method][metric]
                mean_results[method][metric] = float(np.mean(vals)) if vals else 0.0

        save_metrics_csv(mean_results, active_methods, active_metrics, viz_dir)
        plot_metrics_comparison(mean_results, active_methods, active_metrics, viz_dir)
        plot_enhanced_comparison(mean_results, active_methods, active_metrics, viz_dir)
        plot_box_comparison(self.overall_metrics, active_methods, active_metrics, viz_dir)

        print("\nOverall Metrics:")
        for method in active_methods:
            print(f"Method: {method}")
            for metric in active_metrics:
                val = mean_results[method].get(metric, "N/A")
                print(f"  {metric}: {val:.4f}" if isinstance(val, float) else f"  {metric}: {val}")
