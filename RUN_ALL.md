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

### Resultados (144 test, pacientes no vistos; 30 muestras MC/TTA/ruido)

| Método | IoU | Dice | NLL | NLL_fg | ECE | Accuracy | Precision | Recall | Certainty |
|--------|-----|------|-----|--------|-----|----------|-----------|--------|-----------|
| Normal | 0.820 | 0.894 | 0.035 | 0.177 | 0.015 | 0.993 | 0.863 | 0.945 | 0.923 |
| MC Dropout | 0.820 | 0.894 | 0.046 | 0.173 | 0.029 | 0.993 | 0.867 | 0.941 | 0.790 |
| TTA | 0.797 | 0.880 | 0.041 | 0.354 | 0.025 | 0.992 | 0.902 | 0.881 | 0.480 |
| Noisy | 0.820 | 0.894 | 0.035 | 0.171 | 0.015 | 0.993 | 0.863 | 0.945 | 0.894 |
| **Fusion** | **0.826** | **0.899** | 0.035 | 0.171 | 0.019 | 0.993 | 0.873 | 0.942 | 0.859 |
| CRF | 0.827 | 0.900 | 0.035 | 0.171 | 0.019 | 0.993 | 0.875 | 0.941 | 0.833 |

---

## 2. UniVerSeg (canal G - T1c)

Usa solo el canal G (T1c con contraste) replicado 3 veces en vez del RGB completo. El T1c concentra la señal tumoral.

```bash
# 64 support, eval sobre 64 support + 144 test
python -m src.pipelines.run_foundation \
    --config configs/foundation_universeg.yaml \
    --context-size 64
```

### Resultados (G channel, ctx=64; 64 support + 144 test)

| Método | Test Dice | Test IoU | Test ECE | Support Dice | Support IoU |
|--------|:---------:|:--------:|:--------:|:------------:|:-----------:|
| Normal | 0.758 | 0.646 | 0.011 | 0.939 | 0.887 |
| MC Dropout | 0.754 | 0.643 | 0.024 | 0.940 | 0.889 |
| TTA | 0.761 | 0.649 | 0.012 | 0.939 | 0.887 |
| Noisy | **0.766** | **0.655** | 0.011 | 0.939 | 0.887 |
| Fusion | **0.764** | 0.652 | 0.011 | 0.940 | 0.888 |
| CRF | 0.763 | 0.652 | 0.011 | 0.939 | 0.887 |

> TTA para UniVerSeg: 9 transformaciones fotorrométricas — los flips rompen el matching in-context con contexto fijo y ttach Scale choca con el resize interno a 128×128. Support = imágenes ya vistas en contexto; Test = pacientes no vistos.

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
    ├── detailed_metrics.csv           ─ métricas por muestra
    ├── statistical_tests.csv          ─ bootstrap por paciente + Wilcoxon
    ├── enhanced_metrics_comparison.png
    └── box_plot_comparison.png
```

---

## Estadística por paciente

```bash
python -m src.utils.statistics --pipeline all
```

Bootstrap a nivel de paciente (10.000 remuestreos, sin pseudo-replicación de slices) + Wilcoxon sobre medias por paciente. Resultados clave con la ejecución final:

- **UNet**: Fusión vs Normal es significativa (ΔDice **+0.0050**, IC95% [+0.0022, +0.0098], p<0.001, **17/17 pacientes mejoran**); CRF vs Fusión también (+0.0008, p=0.001).
- **UniVerSeg**: Fusión (+0.0058) y Noisy (+0.0079) van en la misma dirección pero no alcanzan significancia con 17 pacientes (IC incluye 0); CRF vs Fusión indistinguible (p=1.0).
