import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from mc_dropout import MCDropout

EPS = 1e-6

class SegCalibratedMCDropout:
    def __init__(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                 p_values: list = [0.05, 0.1, 0.2],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 mc_samples: int = 30, num_classes: int = 2, calib_tolerance: float = 0.02,
                 scale_entropy: bool = False):
        """
        This implementation uses MC dropout and computes uncertainty by combining the 
        expected Hamming distance and its variance from the MC predictions.
        """
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

    @staticmethod
    def hamming_distance(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """
        Computes the Hamming distance (fraction of pixels that differ) between two masks.
        Both masks are assumed to be of shape [B, H, W] with discrete class indices.
        """
        diff = (mask1 != mask2).float()
        return diff.mean().item()

    @staticmethod
    def compute_hamming_stats(mc_preds: torch.Tensor) -> tuple[float, float]:
        """
        Receives a tensor of MC predictions (discrete masks) of shape [T, B, H, W]
        and returns the expectation and variance of the Hamming distances computed over every pair.
        """
        T, B, H, W = mc_preds.shape
        distances = []
        for i in range(T):
            for j in range(i + 1, T):
                dist = SegCalibratedMCDropout.hamming_distance(mc_preds[i], mc_preds[j])
                distances.append(dist)
        distances = np.array(distances)
        expectation = float(np.mean(distances))
        variance = float(np.var(distances))
        return expectation, variance

    def optimize_parameters(self) -> tuple[float, float, float]:
        best_phi, best_scale, best_scale_relaxed = None, None, None
        best_nll = float('inf')
        
        for p in self.p_values:
            self._enable_mc_dropout(p)
            
            # Collect MC samples for loss computation and discrete predictions
            probs_list, mc_preds_list, targets_list, log_probs_list = [], [], [], []

            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)
                    y_indices = y.squeeze(1).long()
                    
                    batch_probs = []
                    batch_preds = []
                    batch_log_probs = []
                    
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        # The output has only one channel with a sigmoid..(Binary Segmentation)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1-logits, logits], dim=1)

                        probs = F.softmax(logits, dim=1)
                        probs = F.softmax(logits, dim=1)
                        batch_probs.append(probs)
                        preds = torch.argmax(probs, dim=1)
                        batch_preds.append(preds.cpu())
                        log_probs = F.log_softmax(logits, dim=1)
                        batch_log_probs.append(log_probs)
                    
                    # [T, B, C, H, W], [T, B, H, W], [T, B, C, H, W]
                    mc_probs = torch.stack(batch_probs)
                    mc_preds = torch.stack(batch_preds)
                    mc_log_probs = torch.stack(batch_log_probs)
                    
                    probs_list.append(mc_probs)
                    mc_preds_list.append(mc_preds)
                    targets_list.append(y_indices)
                    log_probs_list.append(mc_log_probs)
            
            # 1. CORRECT probability averaging across MC samples
            # Concatenate all batches along batch dimension first -> [T, total_B, C, H, W]
            all_probs = torch.cat(probs_list, dim=1)
            avg_probs = torch.mean(all_probs, dim=0)  # [total_B, C, H, W]
            
            # 2. CORRECT entropy calculation
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1)  # [total_B, H, W]

            # Normalize entropy to [0, 1] range in order to make the bucketing more consistent
            if self.scale_entropy:
                entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype=entropy.dtype, device=entropy.device))
            
            # 3. CORRECT target aggregation
            targets = torch.cat(targets_list, dim=0)  # [total_B, H, W]
            
            # 4. CORRECT log probability handling for NLL
            all_log_probs = torch.cat(log_probs_list, dim=1)  # [T, total_B, C, H, W]
            avg_log_probs = torch.mean(all_log_probs, dim=0)  # [total_B, C, H, W]
            nll = F.nll_loss(avg_log_probs, targets).item()
            
            # 5. Hamming statistics
            mc_preds = torch.cat(mc_preds_list, dim=1)  # [T, total_B, H, W]
            exp_hd, var_hd = self.compute_hamming_stats(mc_preds)
            print(f"p={p:.3f} | Exp. Hamming: {exp_hd:.4f} | Var. Hamming: {var_hd:.4f} | NLL: {nll:.4f}")
            
            # Calibration logic
            scale = exp_hd * (1 + var_hd)
            
            try:
                if self.calib_tolerance > 0:
                    scale_relaxed = self._find_relaxed_scale(avg_probs, entropy, targets, scale)
                    print(f"p={p:.3f} | Scale: {scale:.4f} | Relaxed: {scale_relaxed:.4f}")
                else:
                    scale_relaxed = scale
                    print(f"p={p:.3f} | Scale: {scale:.4f}")
                
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

    def _find_relaxed_scale(self, probs: torch.Tensor, entropy: torch.Tensor, 
                            targets: torch.Tensor, init_scale: float) -> float:
        """Calibration-aware scale adjustment using ECE"""
        current_scale = init_scale
        best_ece = float('inf')
        best_scale = init_scale
        
        for _ in range(10):
            scaled_uncertainty = entropy * current_scale
            ece = self.compute_ece(probs, scaled_uncertainty, targets)
            
            if ece < best_ece:
                best_ece = ece
                best_scale = current_scale
                
            current_scale *= 0.95 if ece > self.calib_tolerance else 1.05

        return best_scale

    def compute_ece(self, probs: torch.Tensor, uncertainty: torch.Tensor, 
                    targets: torch.Tensor, n_bins: int = 10) -> float:
        """Compute the Expected Calibration Error (ECE) for uncertainty"""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        bin_indices = torch.bucketize(uncertainty, bin_boundaries)
        
        ece = 0.0
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        targets_flat = targets.flatten()
        
        for bin_idx in range(1, n_bins + 1):
            in_bin = (bin_indices == bin_idx)
            if in_bin.any():
                in_bin_flat = in_bin.flatten()
                bin_conf = (1 - uncertainty[in_bin]).mean()
                bin_acc = (torch.argmax(probs_flat[in_bin_flat], dim=1) == targets_flat[in_bin_flat]).float().mean()
                bin_weight = in_bin_flat.float().mean()
                ece += torch.abs(bin_acc - bin_conf) * bin_weight

        return ece.item() if isinstance(ece, torch.Tensor) else ece

    def compute_uncertainty(self, x: torch.Tensor, use_relaxed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform inference with MC dropout and return predictions and scaled uncertainty."""
        scale = self.best_scale_relaxed if use_relaxed else self.best_scale

        with torch.no_grad():
            mc_probs = []
            for _ in range(self.mc_samples):
                pred = self.model(x)
                # In case of binary segmentation with a sigmoid output
                if pred.shape[1] == 1:
                    pred = torch.cat([1-pred, pred], dim=1)
                
                mc_probs.append(F.softmax(pred, dim=1))

            avg_probs = torch.stack(mc_probs).mean(dim=0)
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1)
            scaled_uncertainty = entropy * scale
        return avg_probs.cpu(), scaled_uncertainty.cpu()

    def _enable_mc_dropout(self, p: float):
        self.mc_dropout.p = p
        self.mc_dropout.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax])

# --------------------------
# Usage Example
# --------------------------
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
        num_classes=2,
        calib_tolerance=0.02
    )

    best_phi, best_scale, best_scale_relaxed = estimator.optimize_parameters()
    print(f"Optimal phi: {best_phi}, Base scale: {best_scale}, Relaxed scale: {best_scale_relaxed}")

    print("Tryout without calibration tolerance:")
    estimator = SegCalibratedMCDropout(
        model=model,
        data_loader=data_loader,
        p_values=[0.05, 0.1, 0.2, 0.3],
        mc_samples=5,
        num_classes=2,
        calib_tolerance=-1.0
    )
    best_phi, best_scale, best_scale_relaxed = estimator.optimize_parameters()
    print(f"Optimal phi: {best_phi}, Base scale: {best_scale}, Relaxed scale: {best_scale_relaxed}")

    test_input = next(iter(data_loader))[0].to(device)
    probs, uncertainty = estimator.compute_uncertainty(test_input, use_relaxed=True)
    print(f"Predictions shape: {probs.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
