To empirically compare the different implementations (using Hamming distance, categorical cross-entropy, and the original NLL-based approach), follow this structured testing plan:

---

### **1. Define Evaluation Metrics**
#### **Segmentation Performance**
- **Dice Coefficient (F1 Score)**: Measures overlap between predictions and ground truth.
- **IoU (Jaccard Index)**: Similar to Dice but penalizes false positives more.
- **Pixel Accuracy**: Fraction of correctly classified pixels.

#### **Uncertainty Calibration**
- **Expected Calibration Error (ECE)**: Quantifies alignment between confidence and accuracy.
- **Brier Score**: Measures probabilistic prediction accuracy (lower is better).
- **Reliability Diagrams**: Visualize confidence vs. accuracy.

#### **Uncertainty Quality**
- **AUROC for Misclassification Detection**: Use uncertainty scores to predict errors.
- **Entropy-Error Correlation**: Correlation between per-pixel entropy and misclassification.

#### **Efficiency**
- **Inference Time**: Time to compute predictions + uncertainty for a batch.
- **Memory Usage**: GPU memory consumption during inference.

---

### **2. Standardized Testing Protocol**
#### **Data Splits**
- Use **three distinct sets**:
  1. **Training Set**: For initial model training (fixed across all methods).
  2. **Validation Set**: For hyperparameter tuning (e.g., `optimize_parameters`).
  3. **Test Set**: For final evaluation (unseen during optimization).

#### **Fixed Hyperparameters**
- Use the same `mc_samples`, `p_values`, and `calib_tolerance` for all methods.
- Ensure identical model architectures and training procedures.

---

### **3. Implementation Steps**
#### **For Each Method** (NLL, Hamming, Cross-Entropy):
1. **Optimize Parameters**:
   - Run `optimize_parameters` on the validation set to find `best_phi`, `best_scale`.
   - Record the optimal parameters for later use.

2. **Test-Set Evaluation**:
   - Compute segmentation metrics (Dice, IoU, Accuracy).
   - Compute calibration metrics (ECE, Brier Score).
   - Calculate AUROC for uncertainty-based error detection.
   - Measure inference time and memory usage.

---

### **4. Statistical Analysis**
- Perform **paired t-tests** or **Wilcoxon signed-rank tests** to assess significant differences in metrics.
- Report mean ± standard deviation for all metrics across multiple runs (if feasible).

---

### **5. Qualitative Evaluation**
- **Visualize Predictions + Uncertainty**:
  - Plot segmentation masks with uncertainty heatmaps for challenging cases.
  - Highlight regions where uncertainty correlates with errors.

---

### **6. Example Code Skeleton**
```python
def evaluate_method(method_class, test_loader, device):
    # Initialize method with pre-trained model
    estimator = method_class(model=model, data_loader=val_loader, ...)
    
    # Optimize parameters on validation set
    best_phi, best_scale, _ = estimator.optimize_parameters()
    
    # Test on held-out test set
    dice_scores, ece_scores, aurocs = [], [], []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        probs, uncertainty = estimator.compute_uncertainty(x)
        
        # Compute metrics
        preds = torch.argmax(probs, dim=1)
        dice = compute_dice(preds, y)
        ece = compute_ece(probs, uncertainty, y)
        auroc = compute_auroc(preds, y, uncertainty)
        
        dice_scores.append(dice)
        ece_scores.append(ece)
        aurocs.append(auroc)
    
    return {
        'Dice': np.mean(dice_scores),
        'ECE': np.mean(ece_scores),
        'AUROC': np.mean(aurocs)
    }

# Compare all methods
methods = {
    'NLL': SegCalibratedMCDropout,
    'Hamming': HammingMCDropoutImplementation,
    'CrossEntropy': CrossEntropyMCDropoutImplementation
}

results = {}
for name, method_class in methods.items():
    results[name] = evaluate_method(method_class, test_loader, device)
```

---

### **7. Expected Output**
A table comparing mean metrics across methods:

| Method          | Dice (%) | ECE ↓ | AUROC (%) | Time (ms) |
|-----------------|----------|-------|-----------|-----------|
| NLL (Original)  | 88.3     | 0.041 | 84.2      | 120       |
| Hamming         | 87.9     | 0.038 | 85.1      | 115       |
| Cross-Entropy   | 89.1     | 0.035 | 86.7      | 125       |

---

### **8. Key Considerations**
- **Reproducibility**: Use fixed random seeds for all experiments.
- **Sensitivity Analysis**: Vary `mc_samples` to assess robustness.
- **Ablation**: Test the impact of `calib_tolerance` and scaling methods.

By following this plan, you’ll systematically identify which method provides the best balance of accuracy, calibration, and efficiency.