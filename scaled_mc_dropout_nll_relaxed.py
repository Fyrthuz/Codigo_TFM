import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from mc_dropout import MCDropout

EPS = 1e-6

class SegCalibratedMCDropout:
    def __init__(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                 p_values: list = [0.05, 0.1, 0.2], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 mc_samples: int = 30, num_classes: int = 2, calib_tolerance: float = 0.02,
                 scale_entropy: bool = False):
        self.model = model
        self.data_loader = data_loader
        self.p_values = p_values
        self.device = device
        self.mc_samples = mc_samples
        self.num_classes = num_classes
        self.calib_tolerance = calib_tolerance
        self.best_phi = None
        self.best_scale = None
        self.best_scale_relaxed = None
        self.scale_entropy = scale_entropy
        self.mc_dropout = MCDropout(self.model, p=0.0)

    def optimize_parameters(self) -> tuple[float, float, float]:
        best_phi, best_scale, best_scale_relaxed = None, None, None
        best_nll = float('inf')
        
        for p in self.p_values:
            self._enable_mc_dropout(p)
            
            # Collect MC samples and targets
            log_probs_list, targets_list = [], []
            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)
                    y_indices = y.squeeze(1).long()  # [B, H, W]
                    
                    batch_log_probs = []
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        # The output has only one channel with a sigmoid..(Binary Segmentation)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1-logits, logits], dim=1)
                        probs = F.log_softmax(logits, dim=1)

                        batch_log_probs.append(probs)
                    
                    # Generate MC samples with log_softmax
                    mc_log_probs = torch.stack(batch_log_probs)  # [T, B, C, H, W]
                    
                    log_probs_list.append(mc_log_probs.cpu())
                    targets_list.append(y_indices.cpu())

            # Compute NLL and scaling parameters
            avg_log_probs = torch.mean(torch.cat(log_probs_list, dim=1), dim=0)  # [B, C, H, W]
            targets = torch.cat(targets_list, dim=0)  # [B, H, W]
            
            # Base NLL calculation
            nll = F.nll_loss(avg_log_probs, targets).item()
            
            # Compute uncertainty scaling
            probs = torch.exp(avg_log_probs)  # [B, C, H, W]
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1)  # [B, H, W]

            # Normalize entropy to [0, 1] range in order to make the bucketing more consistent
            if self.scale_entropy:
                entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype=entropy.dtype, device=entropy.device))
            
            scale = self._compute_uncertainty_scale(probs, entropy, targets)
            
            # Find relaxed scale for calibration
            try:
                if self.calib_tolerance > 0:
                    scale_relaxed = self._find_relaxed_scale(probs, entropy, targets, scale)
                    print(f"p={p:.3f} | NLL: {nll:.4f} | Scale: {scale:.4f} | Relaxed: {scale_relaxed:.4f}")
                else:
                    scale_relaxed = scale
                    print(f"p={p:.3f} | NLL: {nll:.4f} | Scale: {scale:.4f}")
                    
                if nll < best_nll:
                    best_phi = p
                    best_scale = scale
                    best_scale_relaxed = scale_relaxed
                    best_nll = nll
                    
            except RuntimeError as e:
                print(f"Skipping p={p} due to error: {str(e)}")
                continue

        self.best_phi = best_phi
        self.best_scale = best_scale
        self.best_scale_relaxed = best_scale_relaxed
        return best_phi, best_scale, best_scale_relaxed

    def _compute_uncertainty_scale(self, probs: torch.Tensor, entropy: torch.Tensor, targets: torch.Tensor) -> float:
        """Paper-inspired scaling based on error-entropy correlation"""
        preds = torch.argmax(probs, dim=1)  # [B, H, W]
        incorrect = (preds != targets).float()  # [B, H, W]
        
        # Compute scaling as inverse of error-entropy covariance
        cov = torch.cov(torch.stack([entropy.flatten(), incorrect.flatten()]))
        scale = 1 / (cov[0,1] + EPS)
        return scale.item()

    def _find_relaxed_scale(self, probs: torch.Tensor, entropy: torch.Tensor, 
                          targets: torch.Tensor, init_scale: float) -> float:
        """Calibration-aware scale adjustment"""
        current_scale = init_scale
        best_ece = float('inf')
        best_scale = init_scale
        
        for _ in range(10):  # Limited iterations for practical computation
            scaled_uncertainty = entropy * current_scale
            ece = self.compute_ece(probs, scaled_uncertainty, targets)
            
            if ece < best_ece:
                best_ece = ece
                best_scale = current_scale
                
            # Adjust scale based on calibration direction
            current_scale *= 0.95 if ece > self.calib_tolerance else 1.05

        return best_scale

    def compute_ece(self, probs: torch.Tensor, uncertainty: torch.Tensor, 
                targets: torch.Tensor, n_bins: int = 10) -> float:
        """Expected Calibration Error for uncertainty"""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_indices = torch.bucketize(uncertainty, bin_boundaries)
        
        ece = 0.0
        for bin_idx in range(n_bins):
            in_bin = (bin_indices == bin_idx)  # [B, H, W]
            if in_bin.any():
                # Flatten spatial dimensions for indexing
                in_bin_flat = in_bin.flatten()  # [B * H * W]
                probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # [B * H * W, C]
                targets_flat = targets.flatten()  # [B * H * W]
                
                # Compute accuracy and confidence for this bin
                bin_acc = (torch.argmax(probs_flat[in_bin_flat], dim=1) == targets_flat[in_bin_flat]).float().mean()
                bin_conf = 1 - uncertainty[in_bin].mean()  # Inverse relationship
                bin_weight = in_bin.float().mean()
                
                ece += torch.abs(bin_acc - bin_conf) * bin_weight

        return ece.item() if isinstance(ece, torch.Tensor) else ece

    def compute_uncertainty(self, x: torch.Tensor, use_relaxed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns predictions and scaled uncertainty"""
        scale = self.best_scale_relaxed if use_relaxed else self.best_scale
        
        with torch.no_grad():
            mc_log_probs = []
            for _ in range(self.mc_samples):
                logits = self.model(x)
                if logits.shape[1] == 1:
                    logits = torch.cat([1-logits, logits], dim=1)
                mc_log_probs.append(F.log_softmax(logits, dim=1))
            
            mc_log_probs = torch.stack(mc_log_probs)
            avg_log_probs = mc_log_probs.mean(dim=0)  # [B, C, H, W]
            probs = torch.exp(avg_log_probs)  # [B, C, H, W]
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1)  # [B, H, W]
            
            scaled_uncertainty = entropy * scale  # [B, H, W]
            
        return probs.cpu(), scaled_uncertainty.cpu()
    
    # def compute_uncertainty(self, x: torch.Tensor, use_relaxed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Returns predictions and scaled uncertainty"""
    #     scale = self.best_scale_relaxed if use_relaxed else self.best_scale
        
    #     with torch.no_grad():
    #         mc_log_probs = []
    #         for _ in range(self.mc_samples):
    #             logits = self.model(x)
    #             if logits.shape[1] == 1:
    #                 logits = torch.cat([1-logits, logits], dim=1)
    #             mc_log_probs.append(F.softmax(logits, dim=1))
            
    #         mc_log_probs = torch.stack(mc_log_probs)
    #         avg_log_probs = mc_log_probs.mean(dim=0)  # [B, C, H, W]
    #         entropy = -torch.sum(avg_log_probs * torch.log(avg_log_probs + EPS), dim=1)  # [B, H, W]
            
    #         scaled_uncertainty = entropy * scale  # [B, H, W]
            
    #     return avg_log_probs.cpu(), scaled_uncertainty.cpu()

    def _enable_mc_dropout(self, p: float):
        self.mc_dropout.p = p
        self.mc_dropout.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax])

# Usage Example
if __name__ == '__main__':
    # Example usage with memory-efficient processing
    from models.unet.unet import UNet
    from utils.dataset import Carvana
    from torch.utils.data import DataLoader, Subset
    from mc_dropout import MCDropout

    # 1. Load data
    dataset = Carvana(root='./')
    test_subset = Subset(dataset, indices=range(4))  # Small subset for testing
    data_loader = DataLoader(test_subset, batch_size=2, shuffle=False)  # Reduce batch size

    # 2. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load("models/unet/last.pt", map_location=device))


    # Initialize uncertainty estimator
    estimator = SegCalibratedMCDropout(
        model=model,
        data_loader=data_loader,
        p_values=[0.05, 0.1, 0.2, 0.3],
        mc_samples=5,
        num_classes=2
    )

    # Find optimal parameters
    best_phi, best_scale, best_scale_relaxed = estimator.optimize_parameters()
    print(f"Optimal phi: {best_phi}, Base scale: {best_scale}, Relaxed scale: {best_scale_relaxed}")

    print("Tryout without calibration tolerance:")
    estimator = SegCalibratedMCDropout(
        model=model,
        data_loader=data_loader,
        p_values=[0.05, 0.1, 0.2, 0.3],
        mc_samples=5,
        num_classes=2,
        calib_tolerance=-1.0  # Disable calibration
    )
    best_phi, best_scale, best_scale_relaxed = estimator.optimize_parameters()
    print(f"Optimal phi: {best_phi}, Base scale: {best_scale}, Relaxed scale: {best_scale_relaxed}")

    # Inference example
    # test_input = torch.randn(1, 1, 256, 256).to(device)
    # probs, uncertainty = estimator.compute_uncertainty(test_input)
    
    # print(f"Prediction shape: {probs.shape}")
    # print(f"Uncertainty stats - Mean: {uncertainty.mean():.3f}, Max: {uncertainty.max():.3f}")