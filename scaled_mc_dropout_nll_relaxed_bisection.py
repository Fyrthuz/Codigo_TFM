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
                 mc_samples: int = 30, num_classes: int = 2, calib_tolerance: float = 0.1,
                 scale_entropy: bool = False):
        self.model = model
        self.data_loader = data_loader
        self.p_values = p_values
        self.device = device
        self.mc_samples = mc_samples
        self.num_classes = num_classes
        self.calib_tolerance = calib_tolerance  # tolerance for the balance function
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
                        # For binary segmentation, if logits have one channel, convert to two channels.
                        if logits.shape[1] == 1:
                            logits = torch.cat([1 - logits, logits], dim=1)
                        probs = F.log_softmax(logits, dim=1)
                        batch_log_probs.append(probs)
                    
                    mc_log_probs = torch.stack(batch_log_probs)  # [T, B, C, H, W]
                    log_probs_list.append(mc_log_probs.cpu())
                    targets_list.append(y_indices.cpu())

            avg_log_probs = torch.mean(torch.cat(log_probs_list, dim=1), dim=0)  # [B, C, H, W]
            targets = torch.cat(targets_list, dim=0)  # [B, H, W]
            
            # Base NLL calculation (for reporting)
            nll = F.nll_loss(avg_log_probs, targets).item()
            
            # Compute uncertainty as entropy (and scale if requested)
            probs = torch.exp(avg_log_probs)  # [B, C, H, W]
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1)  # [B, H, W]
            if self.scale_entropy:
                entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype=entropy.dtype, device=entropy.device))
            
            # Compute the base scaling factor using NLL loss per pixel.
            # For segmentation, we define the per-pixel "error" as the NLL loss.
            with torch.no_grad():
                # Compute per-pixel NLL loss. This returns a tensor of shape [B, H, W]
                nll_pixels = F.nll_loss(avg_log_probs, targets, reduction='none')
                # Compute ratio per pixel and average to get the scale factor.
                ratio = nll_pixels / (entropy + EPS)
                scale = ratio.mean().item()
            
            # Find relaxed scale for calibration using a bisection method.
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
        self._enable_mc_dropout(self.best_phi)
        return best_phi, best_scale, best_scale_relaxed

    def _compute_balance(self, probs: torch.Tensor, scaled_uncertainty: torch.Tensor, 
                         targets: torch.Tensor, n_bins: int = 10) -> float:
        """
        Computes the calibration balance as the weighted average difference between
        bin accuracy and bin confidence. Here, bin confidence is defined as 1 minus
        the mean scaled uncertainty in the bin. A perfectly calibrated model would have
        a balance of 0.
        """
        # Create bins over the [0, 1] range.
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        # Bucketize the scaled uncertainty.
        bin_indices = torch.bucketize(scaled_uncertainty, bin_boundaries)
        balance = 0.0
        total_weight = 0.0
        
        # Reshape probabilities and targets for pixelwise computation.
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        targets_flat = targets.flatten()
        
        for bin_idx in range(1, n_bins + 1):  # bucket indices start at 1
            in_bin = (bin_indices == bin_idx)
            if in_bin.any():
                in_bin_flat = in_bin.flatten()
                # Compute accuracy in the bin.
                bin_acc = (torch.argmax(probs_flat[in_bin_flat], dim=1) == targets_flat[in_bin_flat]).float().mean()
                # Compute "confidence" as 1 minus the mean scaled uncertainty in the bin.
                bin_conf = 1 - scaled_uncertainty[in_bin].mean()
                # Weight: fraction of pixels in this bin.
                bin_weight = in_bin.float().mean()
                balance += (bin_acc - bin_conf) * bin_weight
                total_weight += bin_weight.item()
                
        if total_weight > 0:
            return balance / total_weight
        else:
            return 0.0

    def _find_relaxed_scale(self, probs: torch.Tensor, entropy: torch.Tensor, 
                            targets: torch.Tensor, init_scale: float) -> float:
        """
        Adjusts the scale factor using a bisection method to find a relaxed scaling factor
        such that the calibration balance (difference between observed and expected confidence)
        is near zero.
        """
        tol = self.calib_tolerance  # tolerance threshold for the balance
        max_iter = 50
        
        # Define an interval around the initial scale.
        c_low = init_scale * 0.5
        c_high = init_scale * 1.5
        
        for i in range(max_iter):
            c_mid = (c_low + c_high) / 2.0
            scaled_uncertainty = entropy * c_mid
            balance_mid = self._compute_balance(probs, scaled_uncertainty, targets, n_bins=10)
            
            # If the balance is within tolerance, return this scale.
            if abs(balance_mid) < tol:
                return c_mid
            
            # Since balance is assumed to be monotonically non-decreasing in c,
            # adjust the interval based on the sign of balance.
            if balance_mid < 0:
                c_low = c_mid
            else:
                c_high = c_mid
        
        return (c_low + c_high) / 2.0

    def compute_ece(self, probs: torch.Tensor, uncertainty: torch.Tensor, 
                    targets: torch.Tensor, n_bins: int = 10) -> float:
        """Expected Calibration Error for uncertainty"""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_indices = torch.bucketize(uncertainty, bin_boundaries)
        
        ece = 0.0
        for bin_idx in range(1, n_bins + 1):
            in_bin = (bin_indices == bin_idx)
            if in_bin.any():
                in_bin_flat = in_bin.flatten()
                probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                targets_flat = targets.flatten()
                bin_acc = (torch.argmax(probs_flat[in_bin_flat], dim=1) == targets_flat[in_bin_flat]).float().mean()
                bin_conf = 1 - uncertainty[in_bin].mean()
                bin_weight = in_bin.float().mean()
                ece += torch.abs(bin_acc - bin_conf) * bin_weight
        return ece.item() if isinstance(ece, torch.Tensor) else ece

    def compute_uncertainty(self, x: torch.Tensor, use_relaxed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns predictions and scaled uncertainty for a given input x"""
        scale = self.best_scale_relaxed if use_relaxed else self.best_scale
        
        with torch.no_grad():
            mc_log_probs = []
            for _ in range(self.mc_samples):
                logits = self.model(x)
                if logits.shape[1] == 1:
                    logits = torch.cat([1 - logits, logits], dim=1)
                mc_log_probs.append(F.log_softmax(logits, dim=1))
            mc_log_probs = torch.stack(mc_log_probs)
            avg_log_probs = mc_log_probs.mean(dim=0)  # [B, C, H, W]
            probs = torch.exp(avg_log_probs)
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1)
            scaled_uncertainty = entropy * scale
        return probs.cpu(), scaled_uncertainty.cpu()
    
    def _enable_mc_dropout(self, p: float):
        self.mc_dropout.p = p
        self.mc_dropout.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax])

# Usage Example (adapt as needed)
if __name__ == '__main__':
    from models.unet.unet import UNet
    from utils.dataset import Carvana
    from torch.utils.data import DataLoader, Subset

    dataset = Carvana(root='./')
    test_subset = Subset(dataset, indices=range(4))
    data_loader = DataLoader(test_subset, batch_size=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load("models/unet/last.pt", map_location=device))

    estimator = SegCalibratedMCDropout(
        model=model,
        data_loader=data_loader,
        p_values=[0.05, 0.1, 0.2, 0.3],
        mc_samples=5,
        num_classes=2
    )

    best_phi, best_scale, best_scale_relaxed = estimator.optimize_parameters()
    print(f"Optimal phi: {best_phi}, Base scale: {best_scale}, Relaxed scale: {best_scale_relaxed}")
