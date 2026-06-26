import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.uncertainty.mc_dropout import MCDropout

EPS = 1e-6


class CalibratedMCDropout:
    def __init__(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
        p_values: list = None, device=None,
        mc_samples: int = 5, num_classes: int = 2, calib_tolerance: float = 0.02,
        scale_entropy: bool = False,
    ):
        self.model = model
        self.data_loader = data_loader
        self.p_values = p_values or [0.1, 0.3, 0.5]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mc_samples = mc_samples
        self.num_classes = num_classes
        self.calib_tolerance = calib_tolerance
        self.best_phi = None
        self.best_scale = None
        self.best_scale_relaxed = None
        self.scale_entropy = scale_entropy
        self.mc_dropout = MCDropout(self.model, p=0.0)

    def _enable_mc_dropout(self, p):
        self.mc_dropout.p = p
        if self.mc_dropout.enabled:
            self.mc_dropout.remove()
        self.mc_dropout.enable(
            ignore_type_layers=[nn.ReLU, nn.Softmax, nn.Sigmoid]
        )

    def _compute_cross_entropy_scale(self):
        log_probs_list, targets_list = [], []
        with torch.no_grad():
            for x, y in tqdm(self.data_loader, desc="Computing scale"):
                x, y = x.to(self.device), y.to(self.device)
                y_indices = y.squeeze(1).long() if y.dim() > 3 else y.long()

                mc_probs = []
                for _ in range(self.mc_samples):
                    logits = self.model(x)
                    if logits.shape[1] == 1:
                        logits = torch.cat([1 - logits, logits], dim=1)
                    probs = F.softmax(logits, dim=1)
                    mc_probs.append(probs)

                mc_probs = torch.stack(mc_probs)
                avg_probs = mc_probs.mean(dim=0)

                log_probs_list.append(avg_probs.cpu())
                targets_list.append(y_indices.cpu())

        avg_probs = torch.cat(log_probs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        B, C, H, W = avg_probs.shape
        probs_flat = avg_probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        ce_loss = F.cross_entropy(probs_flat, targets_flat, reduction="mean").item()
        return ce_loss

    def _calibrate_scale(self, scale, tolerance):
        relaxed_scale = scale
        step = 0.1
        direction = -1
        for _ in range(50):
            test_scale = relaxed_scale + direction * step
            if test_scale <= 0:
                direction *= -1
                step *= 0.5
                continue
            ece = self._compute_ece(test_scale)
            if ece < tolerance:
                relaxed_scale = test_scale
                if direction > 0:
                    step *= 0.9
            else:
                direction *= -1
                step *= 0.5
            if step < 1e-6:
                break
        return relaxed_scale

    def _compute_ece(self, scale):
        self._enable_mc_dropout(self.best_phi)
        all_probs, all_targets = [], []
        with torch.no_grad():
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                if logits.shape[1] == 1:
                    logits = torch.cat([1 - logits, logits], dim=1)
                probs = F.softmax(logits / scale, dim=1)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())
        probs = torch.cat(all_probs, dim=0)
        targets = torch.cat(all_targets, dim=0)

        from src.utils.metrics import compute_ece
        return compute_ece(probs, targets)

    def optimize_parameters(self):
        best_phi, best_scale, best_scale_relaxed = None, None, None
        best_loss = float("inf")

        for p in self.p_values:
            self._enable_mc_dropout(p)
            total_loss = 0.0
            total_batches = 0

            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)
                    y_indices = y.squeeze(1).long() if y.dim() > 3 else y.long()

                    mc_probs = []
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1 - logits, logits], dim=1)
                        probs = F.softmax(logits, dim=1)
                        mc_probs.append(probs)

                    mc_probs = torch.stack(mc_probs).mean(dim=0)
                    B, C, H, W = mc_probs.shape
                    probs_flat = mc_probs.permute(0, 2, 3, 1).reshape(-1, C)
                    targets_flat = y_indices.reshape(-1)
                    loss = F.cross_entropy(probs_flat, targets_flat, reduction="mean")
                    total_loss += loss.item()
                    total_batches += 1

            avg_loss = total_loss / max(total_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_phi = p
                self.best_phi = p

        if best_phi is None:
            best_phi = self.p_values[0]
            self.best_phi = best_phi

        self._enable_mc_dropout(best_phi)
        best_scale = self._compute_cross_entropy_scale()
        best_scale_relaxed = self._calibrate_scale(best_scale, self.calib_tolerance)

        self.best_scale = best_scale
        self.best_scale_relaxed = best_scale_relaxed

        return best_phi, best_scale, best_scale_relaxed
