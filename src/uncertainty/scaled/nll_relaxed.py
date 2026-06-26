import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.uncertainty.mc_dropout import MCDropout

EPS = 1e-6


class SegCalibratedMCDropout:
    def __init__(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
        p_values: list = None, device=None,
        mc_samples: int = 30, num_classes: int = 2, calib_tolerance: float = 0.02,
        scale_entropy: bool = False,
    ):
        self.model = model
        self.data_loader = data_loader
        self.p_values = p_values or [0.05, 0.1, 0.2]
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

    def _compute_nll_scale(self):
        log_probs_list, targets_list = [], []
        with torch.no_grad():
            for x, y in tqdm(self.data_loader, desc="Computing NLL scale"):
                x, y = x.to(self.device), y.to(self.device)
                y_indices = y.squeeze(1).long()

                batch_log_probs = []
                for _ in range(self.mc_samples):
                    logits = self.model(x)
                    if logits.shape[1] == 1:
                        logits = torch.cat([1 - logits, logits], dim=1)
                    probs = F.log_softmax(logits, dim=1)
                    batch_log_probs.append(probs)

                mc_log_probs = torch.stack(batch_log_probs)
                log_probs_list.append(mc_log_probs.cpu())
                targets_list.append(y_indices.cpu())

        avg_log_probs = torch.mean(torch.cat(log_probs_list, dim=1), dim=0)
        targets = torch.cat(targets_list, dim=0)

        B, C, H, W = avg_log_probs.shape
        log_probs_flat = avg_log_probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        nll = F.nll_loss(log_probs_flat, targets_flat, reduction="mean").item()

        mc_log_probs_cat = torch.cat(log_probs_list, dim=1)
        mean_log_probs = mc_log_probs_cat.mean(dim=0)
        centered = mc_log_probs_cat - mean_log_probs.unsqueeze(0)
        T = mc_log_probs_cat.shape[0]
        cov_numerator = centered.permute(1, 2, 3, 0).reshape(-1, T).T @ \
                        centered.permute(1, 2, 3, 0).reshape(-1, T)
        cov = cov_numerator / (B * H * W - 1)

        error = (mean_log_probs.argmax(dim=1) != targets).float().reshape(-1)
        scale = np.abs(np.cov(error.numpy(), cov.diag().numpy())[0, 1]) if cov.numel() > 1 else 1.0
        if scale < EPS:
            scale = 1.0

        return scale

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

    def optimize_parameters(self):
        best_phi, best_scale, best_scale_relaxed = None, None, None
        best_nll = float("inf")

        for p in self.p_values:
            self._enable_mc_dropout(p)

            log_probs_list, targets_list = [], []
            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)
                    y_indices = y.squeeze(1).long()

                    batch_log_probs = []
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1 - logits, logits], dim=1)
                        probs = F.log_softmax(logits, dim=1)
                        batch_log_probs.append(probs)

                    mc_log_probs = torch.stack(batch_log_probs)
                    log_probs_list.append(mc_log_probs.cpu())
                    targets_list.append(y_indices.cpu())

            avg_log_probs = torch.mean(torch.cat(log_probs_list, dim=1), dim=0)
            targets = torch.cat(targets_list, dim=0)

            B, C, H, W = avg_log_probs.shape
            log_probs_flat = avg_log_probs.permute(0, 2, 3, 1).reshape(-1, C)
            targets_flat = targets.reshape(-1)
            nll = F.nll_loss(log_probs_flat, targets_flat, reduction="mean").item()

            if nll < best_nll:
                best_nll = nll
                best_phi = p
                self.best_phi = p

        if best_phi is None:
            best_phi = self.p_values[0]
            self.best_phi = best_phi

        self._enable_mc_dropout(best_phi)
        best_scale = self._compute_nll_scale()
        best_scale_relaxed = self._calibrate_scale(best_scale, self.calib_tolerance)

        self.best_scale = best_scale
        self.best_scale_relaxed = best_scale_relaxed

        return best_phi, best_scale, best_scale_relaxed
