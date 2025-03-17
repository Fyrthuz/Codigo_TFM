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
        self.calib_tolerance = calib_tolerance
        self.best_phi = None
        self.best_scale = None
        self.best_scale_relaxed = None
        self.scale_entropy = scale_entropy
        self.mc_dropout = MCDropout(self.model, p=0.0)

    def optimize_parameters(self) -> tuple[float, float, float]:
        best_phi, best_scale, best_scale_relaxed = None, None, None
        best_ce = float('inf')
        
        for p in self.p_values:
            self._enable_mc_dropout(p)
            
            # Collect MC samples and targets
            logits_list, targets_list = [], []
            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)
                    y_indices = y.squeeze(1).long()
                    
                    batch_logits = []
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1 - logits, logits], dim=1)
                        batch_logits.append(logits)
                    
                    mc_logits = torch.stack(batch_logits)
                    logits_list.append(mc_logits.cpu())
                    targets_list.append(y_indices.cpu())

            avg_logits = torch.mean(torch.cat(logits_list, dim=1), dim=0)  # [B, C, H, W]
            targets = torch.cat(targets_list, dim=0)  # [B, H, W]
            
            # Base CE calculation
            ce_loss = F.cross_entropy(avg_logits, targets).item()
            
            # Compute uncertainty as entropy
            probs = F.softmax(avg_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1)
            if self.scale_entropy:
                entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype=entropy.dtype, device=entropy.device))
            
            # Compute the base scaling factor using CE loss per pixel
            with torch.no_grad():
                ce_pixels = F.cross_entropy(avg_logits, targets, reduction='none')
                ratio = ce_pixels / (entropy + EPS)
                scale = ratio.mean().item()
            
            # Find relaxed scale for calibration
            try:
                if self.calib_tolerance > 0:
                    scale_relaxed = self._find_relaxed_scale(probs, entropy, targets, scale)
                    print(f"p={p:.3f} | CE: {ce_loss:.4f} | Scale: {scale:.4f} | Relaxed: {scale_relaxed:.4f}")
                else:
                    scale_relaxed = scale
                    print(f"p={p:.3f} | CE: {ce_loss:.4f} | Scale: {scale:.4f}")
                    
                if ce_loss < best_ce:
                    best_phi = p
                    best_scale = scale
                    best_scale_relaxed = scale_relaxed
                    best_ce = ce_loss
                    
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
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_indices = torch.bucketize(scaled_uncertainty, bin_boundaries)
        balance = 0.0
        total_weight = 0.0
        
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        targets_flat = targets.flatten()
        
        for bin_idx in range(1, n_bins + 1):
            in_bin = (bin_indices == bin_idx)
            if in_bin.any():
                in_bin_flat = in_bin.flatten()
                bin_acc = (torch.argmax(probs_flat[in_bin_flat], dim=1) == targets_flat[in_bin_flat]).float().mean()
                bin_conf = 1 - scaled_uncertainty[in_bin].mean()
                bin_weight = in_bin.float().mean()
                balance += (bin_acc - bin_conf) * bin_weight
                total_weight += bin_weight.item()
                
        return balance / total_weight if total_weight > 0 else 0.0

    def _find_relaxed_scale(self, probs: torch.Tensor, entropy: torch.Tensor, 
                            targets: torch.Tensor, init_scale: float) -> float:
        tol = self.calib_tolerance
        max_iter = 50
        
        c_low = init_scale * 0.5
        c_high = init_scale * 1.5
        
        for _ in range(max_iter):
            c_mid = (c_low + c_high) / 2.0
            scaled_uncertainty = entropy * c_mid
            balance_mid = self._compute_balance(probs, scaled_uncertainty, targets)
            
            if abs(balance_mid) < tol:
                return c_mid
            
            if balance_mid < 0:
                c_low = c_mid
            else:
                c_high = c_mid
        
        return (c_low + c_high) / 2.0

    def compute_ece(self, probs: torch.Tensor, uncertainty: torch.Tensor, 
                    targets: torch.Tensor, n_bins: int = 10) -> float:
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
        scale = self.best_scale_relaxed if use_relaxed else self.best_scale
        
        with torch.no_grad():
            mc_logits = []
            for _ in range(self.mc_samples):
                logits = self.model(x)
                if logits.shape[1] == 1:
                    logits = torch.cat([1 - logits, logits], dim=1)
                mc_logits.append(logits)
            mc_logits = torch.stack(mc_logits)
            avg_logits = mc_logits.mean(dim=0)
            probs = F.softmax(avg_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1)
            scaled_uncertainty = entropy * scale
        return probs.cpu(), scaled_uncertainty.cpu()
    
    def _enable_mc_dropout(self, p: float):
        self.mc_dropout.p = p
        self.mc_dropout.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax])

# Usage Example remains the same
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