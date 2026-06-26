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

    @staticmethod
    def hamming_distance(mask1, mask2):
        diff = (mask1 != mask2).float()
        return diff.mean().item()

    @staticmethod
    def compute_hamming_stats(mc_preds):
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
        best_hamming = float("inf")

        for p in self.p_values:
            self._enable_mc_dropout(p)

            total_expectation, total_variance = 0.0, 0.0
            num_batches = 0

            with torch.no_grad():
                for x, y in tqdm(self.data_loader, desc=f"Testing p={p}"):
                    x, y = x.to(self.device), y.to(self.device)

                    mc_preds = []
                    for _ in range(self.mc_samples):
                        logits = self.model(x)
                        if logits.shape[1] == 1:
                            logits = torch.cat([1 - logits, logits], dim=1)
                        preds = logits.argmax(dim=1)
                        mc_preds.append(preds)

                    mc_preds = torch.stack(mc_preds)
                    exp, var = self.compute_hamming_stats(mc_preds)
                    total_expectation += exp
                    total_variance += var
                    num_batches += 1

            avg_expectation = total_expectation / max(num_batches, 1)
            avg_variance = total_variance / max(num_batches, 1)

            if avg_expectation < best_hamming:
                best_hamming = avg_expectation
                best_phi = p
                self.best_phi = p

        if best_phi is None:
            best_phi = self.p_values[0]
            self.best_phi = best_phi

        self._enable_mc_dropout(best_phi)

        total_expectation, total_variance = 0.0, 0.0
        num_batches = 0
        with torch.no_grad():
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                mc_preds = []
                for _ in range(self.mc_samples):
                    logits = self.model(x)
                    if logits.shape[1] == 1:
                        logits = torch.cat([1 - logits, logits], dim=1)
                    preds = logits.argmax(dim=1)
                    mc_preds.append(preds)
                mc_preds = torch.stack(mc_preds)
                exp, var = self.compute_hamming_stats(mc_preds)
                total_expectation += exp
                total_variance += var
                num_batches += 1

        avg_expectation = total_expectation / max(num_batches, 1)
        avg_variance = total_variance / max(num_batches, 1)

        best_scale = avg_expectation * (1 + avg_variance)
        if best_scale < EPS:
            best_scale = 1.0

        best_scale_relaxed = self._calibrate_scale(best_scale, self.calib_tolerance)

        self.best_scale = best_scale
        self.best_scale_relaxed = best_scale_relaxed

        return best_phi, best_scale, best_scale_relaxed
