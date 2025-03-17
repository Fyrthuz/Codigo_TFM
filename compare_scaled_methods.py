import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, Subset
from models.unet.unet import UNet
from utils.dataset import Carvana
from scaled_mc_dropout_cross_entropy import CalibratedMCDropout as CrossEntropyMCDropout
from scaled_mc_dropout_nll_relaxed import SegCalibratedMCDropout as NLLMCDropout
from scaled_mc_dropout_hamming import SegCalibratedMCDropout as HammingMCDropout


EPS = 1e-6
SAVE_DIR = './results'

# -------------------------
# 📌 1. Función para Calcular ECE
# -------------------------
def compute_ece(probs, targets, n_bins=10):
    """
    Computes ECE by binning based on confidence (max probability).
    """
    # Compute confidence (max probability) and predictions
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])  # [B*H*W, C]
    confidences = probs_flat.max(dim=1)[0]
    predictions = probs_flat.argmax(dim=1)
    targets_flat = targets.flatten()

    # Bin boundaries for confidence [0, 1]
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_indices = torch.bucketize(confidences, bin_boundaries, right=True)

    ece = 0.0
    for bin_idx in range(1, n_bins + 1):
        in_bin = bin_indices == bin_idx
        if in_bin.any():
            bin_acc = (predictions[in_bin] == targets_flat[in_bin]).float().mean()
            bin_conf = confidences[in_bin].mean()
            bin_weight = in_bin.float().mean()
            ece += torch.abs(bin_acc - bin_conf) * bin_weight

    return ece.item()
# -------------------------
# 📌 2. Función para Calcular NLL
# -------------------------
def compute_nll(probs, targets):
    """
    Calcula la pérdida de Log-Negativa (NLL).
    """
    print(probs.min(),probs.max(), probs.shape)
    print(targets.min(), targets.max(), targets.shape)
    log_probs = torch.log(probs + EPS)  # Convertir a log-probabilidad
    return F.nll_loss(log_probs, targets).item()

# -------------------------
# 📌 3. Función para Calcular Brier Score
# -------------------------
def compute_brier_score(probs, targets, num_classes):
    """
    Calcula el Brier Score para segmentación.
    """
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    return torch.mean((probs - targets_one_hot) ** 2).item()

# -------------------------
# 📌 4. Función para Calcular Dice Coefficient (DSC)
# -------------------------
def compute_dice_score(preds, targets):
    """
    Calcula el Dice Score Coefficient (DSC) entre predicciones y etiquetas.
    """
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    return (2.0 * intersection / (union + EPS)).item()

# -------------------------
# 📌 5. Función para Calcular IoU
# -------------------------
def compute_iou(preds, targets):
    """
    Calcula Intersection over Union (IoU).
    """
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets) - intersection
    return (intersection / (union + EPS)).item()

# -------------------------
# 📌 6. Función para Calcular Accuracy
# -------------------------
def compute_accuracy(preds, targets):
    """
    Calcula la precisión total (pixel-wise accuracy).
    """
    return (torch.sum(preds == targets) / targets.numel()).item()

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-6

# -------------------------
# 📌 7. Función para Optimizar los Parámetros de Escala
# -------------------------
def optimize_and_evaluate(methods, dataloader, num_classes, device):
    results = {name: {"ECE": 0, "NLL": 0, "Brier": 0, "Dice": 0, "IoU": 0, "Accuracy": 0} for name, _ in methods}
    total_batches = 0

    optimized_methods = {}
    
    for name, method in methods:
        print(f"\n🔹 Optimizing {name}...")
        best_phi, best_scale, best_scale_relaxed = method.optimize_parameters()
        optimized_methods[name] = (method, best_scale_relaxed)
        
        # Create directories for each method
        save_base = SAVE_DIR
        method_dir = os.path.join(save_base, name)
        os.makedirs(os.path.join(method_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(method_dir, 'segmentation'), exist_ok=True)
        os.makedirs(os.path.join(method_dir, 'uncertainty'), exist_ok=True)
        os.makedirs(os.path.join(method_dir, 'ground_truth'), exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_indices = y.squeeze(1).long()

            for name, (method, best_scale_relaxed) in optimized_methods.items():
                probs, uncertainty = method.compute_uncertainty(x, use_relaxed=True)
                # Ensure probs and uncertainty are on the correct device
                probs = probs.to(device)
                uncertainty = uncertainty.to(device)
                preds = torch.argmax(probs, dim=1)

                # Calculate metrics
                results[name]["ECE"] += compute_ece(probs, y_indices)
                results[name]["NLL"] += compute_nll(probs, y_indices)
                results[name]["Brier"] += compute_brier_score(probs, y_indices, num_classes)
                results[name]["Dice"] += compute_dice_score(preds, y_indices)
                results[name]["IoU"] += compute_iou(preds, y_indices)
                results[name]["Accuracy"] += compute_accuracy(preds, y_indices)

                # Save images for each sample in the batch
                batch_size = x.size(0)
                for i in range(batch_size):
                    sample_idx = batch_idx * dataloader.batch_size + i
                    
                    # Original Image (convert from tensor to numpy)
                    original = x[i].cpu().numpy().transpose(1, 2, 0)
                    original = (original * 255).astype(np.uint8)
                    original_path = os.path.join(SAVE_DIR, name, 'original', f'sample_{sample_idx}.png')
                    plt.imsave(original_path, original)
                    
                    # Segmentation Mask
                    seg = preds[i].cpu().numpy().astype(np.uint8) * 255
                    seg_path = os.path.join(SAVE_DIR, name, 'segmentation', f'sample_{sample_idx}.png')
                    plt.imsave(seg_path, seg, cmap='gray')
                    
                    # Uncertainty Map
                    uncert = uncertainty[i].cpu().numpy()
                    uncert_path = os.path.join(SAVE_DIR, name, 'uncertainty', f'sample_{sample_idx}.png')
                    plt.imsave(uncert_path, uncert, cmap='viridis')
                    
                    # Ground Truth
                    gt = y_indices[i].cpu().numpy().astype(np.uint8) * 255
                    gt_path = os.path.join(SAVE_DIR, name, 'ground_truth', f'sample_{sample_idx}.png')
                    plt.imsave(gt_path, gt, cmap='gray')

            total_batches += 1

    # Average metrics over all batches
    for name in results:
        for metric in results[name]:
            results[name][metric] /= total_batches

    return results
# -------------------------
# 📌 8. Función para Graficar Comparaciones
# -------------------------
def plot_results(results):
    """
    Genera gráficos de comparación entre métodos.
    """
    methods = list(results.keys())
    metrics = ["ECE", "NLL", "Brier", "Dice", "IoU", "Accuracy"]
    colors = ["blue", "orange", "green"]

    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i + 1)
        values = [results[m][metric] for m in methods]
        plt.bar(methods, values, color=colors)
        plt.ylabel(metric)
        plt.title(f"Comparación de {metric}")
        plt.tight_layout()

    plt.tight_layout()
    # plt.show()
    plt.savefig(SAVE_DIR + "/results.png")

# -------------------------
NUM_SAMPLES = 8
BATCH_SIZE = 4
P_VALUES = [0.001 * i for i in range(1,2)]
MC_SAMPLES = 5
NUM_CLASSES = 2

# 1️⃣ Cargar Datos
dataset = Carvana(root='./')
test_subset = Subset(dataset, indices=range(NUM_SAMPLES))  # Subset de validación
data_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

# 2️⃣ Inicializar el Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=2).to(device)
model.load_state_dict(torch.load("models/unet/last.pt", map_location=device))

# 3️⃣ Inicializar las Implementaciones
method1 = HammingMCDropout(model, data_loader, p_values=P_VALUES, mc_samples=MC_SAMPLES, num_classes=NUM_CLASSES)
method2 = NLLMCDropout(model, data_loader, p_values=P_VALUES, mc_samples=MC_SAMPLES, num_classes=NUM_CLASSES)
method3 = CrossEntropyMCDropout(model, data_loader, p_values=P_VALUES, mc_samples=MC_SAMPLES, num_classes=NUM_CLASSES)

# 4️⃣ Evaluar las Implementaciones
methods = [
    ("Hamming Distance", method1),
    ("Error-Entropy Covariance", method2),
    ("Cross-Entropy Loss", method3),
]

results = optimize_and_evaluate(methods, data_loader, num_classes=2, device=device)

# 5️⃣ Mostrar los Resultados
plot_results(results)

# Print results
for name, metrics in results.items():
    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
