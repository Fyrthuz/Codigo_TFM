import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mc_dropout import MCDropout

EPS = 1e-6

class CalibratedMCDropout:
    def __init__(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                 p_values: list = [0.1, 0.3, 0.5],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 mc_samples: int = 5, num_classes: int = 2, calib_tolerance: float = 0.02,
                 scale_entropy: bool = False):
        """
        This class evaluates different dropout probabilities (phi) using Monte Carlo dropout.
        It uses categorical cross entropy (applied on averaged probabilities) as loss,
        and then computes an uncertainty scale based on the error-entropy covariance.
        A “relaxed” scale is further obtained by iteratively adjusting the scale to reduce
        the Expected Calibration Error (ECE) below a given tolerance.
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

    def optimize_parameters(self) -> tuple[float, float, float]:
        best_phi, best_scale, best_scale_relaxed = None, None, None
        best_loss = float('inf')
        
        for p in self.p_values:
            # Set dropout probability for all Dropout layers
            self._enable_mc_dropout(p)
            
            total_loss = 0.0
            total_batches = 0

            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)
                    # For segmentation, y is expected to be of shape [B, H, W]
                    y_indices = y.squeeze(1).long() if y.dim() > 3 else y.long()
                    
                    # Obtain MC samples: run several forward passes and collect softmax probabilities
                    mc_probs = []
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        # The output has only one channel with a sigmoid..(Binary Segmentation)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1-logits, logits], dim=1)
                        probs = F.softmax(logits, dim=1)
                        mc_probs.append(probs)
                    avg_probs = torch.stack(mc_probs).mean(dim=0)  # shape: (B, C, H, W)
                    
                    # Compute loss using categorical cross entropy on averaged probabilities.
                    # Since cross entropy expects log-probabilities, we convert avg_probs using log.
                    avg_log_probs = torch.log(avg_probs + EPS)
                    loss = F.nll_loss(avg_log_probs, y_indices)
                    total_loss += loss.item()
                    total_batches += 1

            avg_loss = total_loss / total_batches

            # Now compute the uncertainty scaling parameters.
            # We obtain predictions and entropy from the averaged probabilities.
            preds = torch.argmax(avg_probs, dim=1)
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1)  # (B, H, W)

            # Normalize entropy to [0, 1] range in order to make the bucketing more consistent
            if self.scale_entropy:
                entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype=entropy.dtype, device=entropy.device))

            # Compute a basic scale as the inverse of the covariance between error and entropy.
            incorrect = (preds != y_indices).float()
            entropy_flat = entropy.flatten()
            incorrect_flat = incorrect.flatten()
            if entropy_flat.numel() < 2:
                cov = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            else:
                cov = torch.cov(torch.stack([entropy_flat, incorrect_flat]))
            scale = 1 / (cov[0, 1] + EPS)
            scale = scale.item()

            # Compute the "relaxed" scale by iteratively adjusting the scale to lower ECE.
            if self.calib_tolerance > 0:
                scale_relaxed = self._find_relaxed_scale(avg_probs, entropy, y_indices, scale)
                print(f"p={p:.3f} | Loss: {avg_loss:.4f} | Scale: {scale:.4f} | Relaxed: {scale_relaxed:.4f}")
            else:
                scale_relaxed = scale
                print(f"p={p:.3f} | Loss: {avg_loss:.4f} | Scale: {scale:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_phi = p
                best_scale = scale
                best_scale_relaxed = scale_relaxed

        self.best_phi = best_phi
        self.best_scale = best_scale
        self.best_scale_relaxed = best_scale_relaxed
        return best_phi, best_scale, best_scale_relaxed

    def _find_relaxed_scale(self, probs: torch.Tensor, entropy: torch.Tensor, 
                            targets: torch.Tensor, init_scale: float) -> float:
        """Calibration-aware scale adjustment based on expected calibration error (ECE)"""
        current_scale = init_scale
        best_ece = float('inf')
        best_scale = init_scale

        for _ in range(10):  # Limited number of iterations
            scaled_uncertainty = entropy * current_scale
            ece = self.compute_ece(probs, scaled_uncertainty, targets)
            
            if ece < best_ece:
                best_ece = ece
                best_scale = current_scale

            # Adjust the scale: if ECE is too high, decrease scale; if low, increase it.
            current_scale *= 0.95 if ece > self.calib_tolerance else 1.05

        return best_scale

    def compute_ece(self, probs: torch.Tensor, uncertainty: torch.Tensor, 
                    targets: torch.Tensor, n_bins: int = 10) -> float:
        """
        Compute the Expected Calibration Error (ECE) for the uncertainty.
        Assumes segmentation output: probs shape (B, C, H, W) and targets shape (B, H, W).
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        bin_indices = torch.bucketize(uncertainty, bin_boundaries)

        ece = 0.0
        # Flatten spatial dimensions for evaluation.
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        targets_flat = targets.flatten()

        for bin_idx in range(1, n_bins + 1):
            in_bin = (bin_indices == bin_idx)
            if in_bin.any():
                in_bin_flat = in_bin.flatten()
                # For calibration we compare the confidence (inverse uncertainty) with accuracy.
                bin_conf = (1 - uncertainty[in_bin]).mean()
                bin_acc = (torch.argmax(probs_flat[in_bin_flat], dim=1) == targets_flat[in_bin_flat]).float().mean()
                bin_weight = in_bin_flat.float().mean()
                ece += torch.abs(bin_acc - bin_conf) * bin_weight
        
        return ece.item() if isinstance(ece, torch.Tensor) else ece

    def compute_uncertainty(self, x: torch.Tensor, use_relaxed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the averaged probabilities (predictions) and the scaled uncertainty map.
        If use_relaxed is True, the relaxed calibration scale is used.
        """
        scale = self.best_scale_relaxed if use_relaxed else self.best_scale

        with torch.no_grad():
            mc_probs = []
            for _ in range(self.mc_samples):
                logits = self.model(x)
                if logits.shape[1] == 1:
                    logits = torch.cat([1-logits, logits], dim=1)
                mc_probs.append(F.softmax(logits, dim=1))
            avg_probs = torch.stack(mc_probs).mean(dim=0)
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS), dim=1)
            scaled_uncertainty = entropy * scale

        return avg_probs.cpu(), scaled_uncertainty.cpu()

    def _enable_mc_dropout(self, p: float):
        self.mc_dropout.p = p
        self.mc_dropout.enable(ignore_type_layers=[torch.nn.ReLU, torch.nn.Softmax])

if __name__ == '__main__':
    # Example usage (using UNet for segmentation and Carvana dataset as in the original code)
    from models.unet.unet import UNet
    from utils.dataset import Carvana
    from torch.utils.data import DataLoader, Subset

    # 1. Load data
    dataset = Carvana(root='./')
    test_subset = Subset(dataset, indices=range(4))  # Small subset for testing
    data_loader = DataLoader(test_subset, batch_size=2, shuffle=False)

    # 2. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load("models/unet/last.pt", map_location=device))

    # 3. Initialize the estimator with MC Dropout calibration
    estimator = CalibratedMCDropout(
        model=model,
        data_loader=data_loader,
        p_values=[0.1, 0.3, 0.5],
        mc_samples=5,
        num_classes=2,
        calib_tolerance=0.02
    )

    # 4. Optimize dropout rate (phi) and obtain scales
    best_phi, best_scale, best_scale_relaxed = estimator.optimize_parameters()
    print(f"Optimal phi: {best_phi}, Base scale: {best_scale}, Relaxed scale: {best_scale_relaxed}")


    print("Tryout without calibration tolerance:")
    estimator = CalibratedMCDropout(
        model=model,
        data_loader=data_loader,
        p_values=[0.1, 0.3, 0.5],
        mc_samples=5,
        num_classes=2,
        calib_tolerance=-1.0  # Disable calibration
    )
    best_phi, best_scale, best_scale_relaxed = estimator.optimize_parameters()

    print(f"Optimal phi: {best_phi}, Base scale: {best_scale}, Relaxed scale: {best_scale_relaxed}")

    # 5. Compute uncertainty for a test batch
    test_input = next(iter(data_loader))[0].to(device)
    probs, uncertainty = estimator.compute_uncertainty(test_input, use_relaxed=True)
    print(f"Predictions shape: {probs.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
