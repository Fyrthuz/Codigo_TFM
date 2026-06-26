# Ejecución Completa de Pipelines

## Requisitos previos

```bash
source .venv/bin/activate
./scripts/download_data.sh lgg       # Kaggle → MRI/filtered_data/ (1.060 imágenes, 1% threshold)
pip install git+https://github.com/JJGO/UniverSeg.git
```

---

## 1. UNet 2D

### Entrenamiento
- Split paciente-nivel 70/15/15 (~750/160/160 imágenes, sin fuga)
- UNet init_features=32, BCE+Dice loss, Adam lr=1e-4, batch=16
- Augmentation: flips, rot ±20°, scale ±10%, color jitter
- Early stopping, 60 epochs máx. Best Val IoU = 0.83

```bash
python -m src.utils.train_unet --data-root ./MRI/filtered_data --epochs 60
python -m src.pipelines.run_unet --config configs/pipeline_2d.yaml --checkpoint unet_model.pth
```

### Resultados (144 test, pacientes no vistos)

| Método | IoU | Dice | NLL | Accuracy | Precision | Recall | Certainty |
|--------|-----|------|-----|----------|-----------|--------|-----------|
| Normal | 0.820 | 0.894 | 0.035 | 0.993 | 0.863 | 0.945 | 0.923 |
| MC Dropout | 0.819 | 0.894 | 0.046 | 0.993 | 0.867 | 0.941 | 0.799 |
| TTA | 0.797 | 0.880 | 0.041 | 0.992 | 0.902 | 0.881 | 0.480 |
| Noisy | 0.820 | 0.894 | 0.035 | 0.993 | 0.863 | 0.946 | 0.894 |
| **Fusion** | **0.826** | **0.899** | 0.035 | 0.993 | 0.873 | 0.942 | 0.862 |
| CRF | 0.511 | 0.606 | 0.515 | 0.977 | 0.914 | 0.527 | 0.361 |

---

## 2. UniVerSeg (canal G - T1c)

Usa solo el canal G (T1c con contraste) replicado 3 veces en vez del RGB completo. El T1c concentra la señal tumoral.

```bash
# 64 support de train, eval sobre ~160 test
python -m src.pipelines.run_foundation \
    --config configs/foundation_universeg.yaml \
    --context-size 64
```

### Resultados (G channel, ctx=64)

| Método | Test Dice | Test IoU | Support Dice | Support IoU |
|--------|:---------:|:--------:|:------------:|:-----------:|
| Normal | **0.762** | 0.651 | **0.948** | 0.903 |
| MC Dropout | 0.760 | 0.649 | 0.945 | 0.901 |
| TTA | — | — | — | — |
| Noisy | **0.776** | 0.665 | 0.949 | 0.904 |
| Fusion | **0.776** | 0.665 | 0.949 | 0.905 |
| CRF | 0.342 | 0.212 | 0.438 | 0.296 |

> TTA no disponible (conflicto con resize interno 128×128). Support = imágenes ya vistas en contexto; Test = pacientes no vistos.

---

## Estructura de resultados

```
results/ (UNet) o results_foundation_universeg/ (UniVerSeg)
├── sample_0/ (o support_0/, test_0/)
│   ├── original_image.png
│   ├── ground_truth.png
│   ├── original/         probability.png, mask.png, uncertainty.png
│   ├── mc_dropout/       mean_prediction.png, uncertainty.png, predictions/
│   ├── tta/
│   ├── noisy/
│   ├── fusion/           probability.png, mask.png, uncertainty.png
│   └── refined/          CRF result
└── visualizations/
    ├── metrics_summary.csv
    ├── enhanced_metrics_comparison.png
    └── box_plot_comparison.png
```
