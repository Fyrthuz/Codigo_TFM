# TFM: Medical Image Segmentation with Uncertainty Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)

Master's Thesis comparing **UNet** and **UniVerSeg** with **uncertainty quantification** (MC Dropout, TTA, Noisy Inference, Fusion, CRF) on brain MRI (LGG Segmentation Dataset). **Patient-level split** ensures no data leakage between train/test.

---

## Features

- **UNet 2D** — trained from scratch (60 epochs, augmentation, early stopping)
- **UniVerSeg few-shot** — zero-shot with configurable context size (1–128 images)
- **Uncertainty methods** — MC Dropout, TTA, Noisy, Fusion, CRF
- **Pure numpy/OpenCV CRF** — no compilation needed
- **Patient-level split** — 70/15/15 over 108 patients, ~160 test images

## Dataset

**LGG MRI Segmentation** (Kaggle): 3,929 images from 110 patients. RGB channels = T1/T1c/FLAIR. Filtered at **1% foreground threshold** → **1,060 images from 108 patients** (removes slices with <1% tumor while retaining 77% of tumor images).

```bash
./scripts/download_data.sh lgg
```

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
uv pip install -e .
pip install git+https://github.com/JJGO/UniverSeg.git

# Download + filter dataset
./scripts/download_data.sh lgg

# UNet: train + evaluate
python -m src.utils.train_unet --data-root ./MRI/filtered_data --epochs 60
python -m src.pipelines.run_unet --config configs/pipeline_2d.yaml --checkpoint unet_model.pth

# UniVerSeg: few-shot
python -m src.pipelines.run_foundation --config configs/foundation_universeg.yaml --context-size 64
```

## Results

### UNet 2D

| Evaluación | IoU | Dice | NLL | Acc | Precision | Recall |
|:----------|:---:|:----:|:---:|:---:|:---------:|:------:|
| **Test set** (144 img, 1-7% tumor, pacientes no vistos) | 0.820 | 0.894 | 0.035 | 0.993 | 0.863 | 0.945 |
| **Solo >7% tumor** (95 img, tumores grandes) | **0.881** | **0.935** | — | — | — | — |

> **Nota**: La métrica oficial (Dice 0.894) usa **split por paciente** y **todos los tamaños de tumor (1-7%)**. Evaluando solo sobre tumores >7% (95 imágenes) se obtiene Dice 0.935.

| Método de incertidumbre | Dice | IoU | NLL | Accuracy | Certainty |
|------------------------|:----:|:---:|:---:|:--------:|:---------:|
| Normal | 0.894 | 0.820 | 0.035 | 0.993 | 0.923 |
| MC Dropout | 0.894 | 0.819 | 0.046 | 0.993 | 0.799 |
| TTA | 0.880 | 0.797 | 0.041 | 0.992 | 0.480 |
| Noisy | 0.894 | 0.820 | 0.035 | 0.993 | 0.894 |
| **Fusion** | **0.899** | **0.826** | 0.035 | 0.993 | 0.862 |
| CRF | 0.606 | 0.511 | 0.515 | 0.977 | 0.361 |

### UniVerSeg (G channel T1c, context-size 64)

| Método de incertidumbre | Test Dice | Test IoU | Support Dice | Support IoU |
|------------------------|:---------:|:--------:|:------------:|:-----------:|
| **Normal** | **0.762** | 0.651 | **0.948** | 0.903 |
| **MC Dropout** | 0.760 | 0.649 | 0.945 | 0.901 |
| **TTA** | — | — | — | — |
| **Noisy** | **0.776** | 0.665 | 0.949 | 0.904 |
| **Fusion** | **0.776** | 0.665 | 0.949 | 0.905 |
| **CRF** | 0.342 | 0.212 | 0.438 | 0.296 |

> TTA no disponible (conflicto con resize interno a 128×128). Canal G (T1c) usado en lugar de RGB completo — ver estudio abajo.

> UniVerSeg con canal G (T1c) alcanza el **85% del rendimiento de UNet sin necesidad de entrenamiento**. Sobre las imágenes de soporte (que ya ha visto en contexto), iguala a UNet (Dice 0.948 vs 0.894).

---

## Studies

### 1. Impact of foreground threshold on metrics

Model trained at 1% threshold, evaluated on increasingly strict subsets of the test set:

| Threshold | Test imgs | UNet Dice | UniVerSeg Dice |
|:---------:|:---------:|:---------:|:--------------:|
| 1% | 144 | 0.898 | 0.416 |
| 2% | 104 | **0.925** | 0.459 |
| 3% | 72 | **0.927** | 0.482 |
| 4% | 50 | 0.917 | **0.532** |
| 5% | 28 | 0.902 | 0.551 |
| 7% | 15 | 0.876 | **0.587** |

- **UNet**: Peaks at 2-3% threshold (Dice 0.927). Declines past 4% due to training data scarcity at higher thresholds.
- **UniVerSeg**: Improves monotonically as threshold increases. Larger tumors are inherently easier for few-shot models.

### 2. UniVerSeg: context-size impact (RGB input, re-evaluar con G channel)

Con entrada RGB completa (3 canales promediados):

| Context size | Support Dice | Test Dice | Gap |
|:-----------:|:-----------:|:---------:|:---:|
| 1 | 0.941 | 0.135 | 0.806 |
| 2 | 0.953 | 0.153 | 0.800 |
| 4 | 0.944 | 0.169 | 0.775 |
| 8 | 0.941 | 0.158 | 0.783 |
| 16 | 0.944 | 0.279 | 0.665 |
| 32 | 0.929 | 0.315 | 0.614 |
| 64 | 0.896 | 0.416 | 0.480 |
| **128** | **0.846** | **0.576** | 0.270 |

> Con canal G (T1c) el rendimiento mejora significativamente: ctx=64 alcanza **Dice 0.762** en test. Ver estudio #3.

### 3. UniVerSeg: input channel impact (RGB vs G channel)

UniVerSeg convierte entrada a grises promediando 3 canales. El canal G (T1c, con contraste) concentra la información tumoral:

| Entrada | Canales promediados | Test Dice |
|:-------:|:-------------------:|:---------:|
| **RGB completo** | (T1 + T1c + FLAIR) / 3 | 0.416 |
| R channel ×3 (T1) | (T1 + T1 + T1) / 3 = T1 | 0.430 |
| **G channel ×3 (T1c)** | **(T1c + T1c + T1c) / 3 = T1c** | **0.762** |
| B channel ×3 (FLAIR) | (FLAIR + FLAIR + FLAIR) / 3 = FLAIR | 0.358 |
| Grayscale avg | (T1 + T1c + FLAIR) / 3 | 0.647 |

> **Conclusión**: Usar solo el canal G (T1c) mejora el Dice de 0.416 a **0.762 (+83%)**. El contraste de T1c resalta los tumores; al promediarlo con T1 y FLAIR se diluye la señal.

### 4. UniVerSeg: same vs unseen images

| Context | Dice (same images) | Dice (unseen test) | Gap narrows as context grows |
|:-------:|:-----------------:|:------------------:|:---------------------------:|
| 8 | 0.941 | 0.158 | 0.783 |
| 64 | 0.896 | 0.416 | 0.480 |
| 128 | 0.846 | 0.576 | 0.270 |

More context examples act as regularization: the model learns broader patterns instead of memorizing individual examples.

---

## Output Structure

Cada pipeline genera resultados en su directorio (`results/` para UNet, `results_foundation_universeg/` para UniVerSeg):

```
results/
├── sample_0/                     (o support_0/, test_0/ para UniVerSeg)
│   ├── original_image.png        ─ imagen de entrada
│   ├── ground_truth.png          ─ máscara real
│   ├── original/                 ─ inferencia normal
│   │   ├── probability.png       ─ mapa de probabilidad
│   │   ├── mask.png              ─ máscara binaria (>0.5)
│   │   └── uncertainty.png       ─ mapa de incertidumbre (1 - prob)
│   ├── mc_dropout/               ─ MC Dropout (10 pasadas)
│   │   ├── mean_prediction.png
│   │   ├── uncertainty.png       ─ entropía de las predicciones
│   │   └── predictions/          ─ las 10 máscaras individuales
│   ├── tta/                      ─ Test-Time Augmentation (9 transforms)
│   ├── noisy/                    ─ 10 pasadas con ruido gaussiano
│   ├── fusion/                   ─ media ponderada por incertidumbre
│   └── refined/                  ─ CRF sobre la fusión
└── visualizations/
    ├── metrics_summary.csv       ─ media de todas las métricas por método
    ├── detailed_metrics.csv      ─ métricas por muestra individual
    ├── metrics_summary.png       ─ gráfico de barras
    ├── enhanced_metrics_comparison.png
    └── box_plot_comparison.png
```

## Uncertainty Methods

| Method | Description | UNet | UniVerSeg |
|--------|-------------|:----:|:---------:|
| **Normal** | Single forward pass | ✓ | ✓ |
| **MC Dropout** | 10 passes with random dropout (p=0.01) on all layers | ✓ | ✓ |
| **TTA** | 9 transforms (horizontal flip, scale ×3, multiply ×5) + average | ✓ | ✗* |
| **Noisy** | 10 passes with Gaussian noise (σ=0.01) added to input | ✓ | ✓ |
| **Fusion** | Uncertainty-weighted average of MC+TTA+Noisy (inverse weighting) | ✓ | ✓ |
| **CRF** | Dense CRF refinement (numpy/OpenCV, 5 iterations) | ✓ | ✓ |

> *TTA incompatible with UniVerSeg's internal 128×128 resize (Scale transform produces varying sizes).

## Project Structure

```
├── src/
│   ├── config.py                  ─ Configuración YAML → dataclasses (PipelineConfig)
│   ├── models/
│   │   ├── unet.py                ─ UNet 2D (Conv2d, BatchNorm2d, up-conv)
│   │   ├── dense_nn.py            ─ MNIST demo classifier
│   │   └── foundation/
│   │       ├── base.py            ─ FoundationModel (ABC, NoTrainingRequired mixin)
│   │       └── universeg.py       ─ UniVerSeg few-shot wrapper (canal G por defecto)
│   ├── pipelines/
│   │   ├── base.py                ─ BaseSegmentationPipeline (toda la lógica de incertidumbre)
│   │   ├── unet.py                ─ UNetPipeline (carga test_indices.json para evaluar solo test)
│   │   ├── foundation.py          ─ FoundationPipeline (evalúa support + test por separado)
│   │   ├── run_unet.py            ─ Entry point UNet
│   │   └── run_foundation.py      ─ Entry point UniVerSeg (carga canal G)
│   ├── uncertainty/
│   │   ├── mc_dropout.py          ─ MCDropout wrapper + mc_dropout_inference()
│   │   ├── tta.py                 ─ tta_inference() con ttach
│   │   └── noise_inference.py     ─ NoisyInference + noisy_inference()
│   └── utils/
│       ├── metrics.py             ─ compute_iou, dice, metrics (NLL, ECE, Brier...)
│       ├── fusion.py              ─ weighted_average_with_uncertainty()
│       ├── crf.py                 ─ Dense CRF (numpy/OpenCV, log-space)
│       ├── visualization.py       ─ save_image, plot_metrics_comparison, box plots
│       ├── dataset.py             ─ LGGSegmentationDataset + split_by_patient()
│       ├── filter_data_mri.py     ─ Filtrado por foreground ratio (default 1%)
│       ├── train_unet.py          ─ Training loop (split paciente-nivel, augmentation, early stopping)
│       └── download_datasets.py   ─ Descarga LGG desde Kaggle + filtrado
├── tests/                         ─ 53 tests (pytest)
│   ├── test_models.py, test_mc_dropout.py, test_tta.py, test_noise.py
│   ├── test_metrics.py, test_fusion.py, test_crf.py
│   ├── test_datasets.py, test_config.py, test_foundation_models.py
├── configs/
│   ├── pipeline_2d.yaml           ─ Config UNet (paths, inferencia, fusión, CRF)
│   └── foundation_universeg.yaml  ─ Config UniVerSeg
├── scripts/
│   ├── download_data.sh           ─ ./scripts/download_data.sh lgg
│   ├── run_pipeline_2d.sh         ─ UNet pipeline (entrenar + evaluar)
│   └── run_foundation.sh          ─ UniVerSeg pipeline
└── MRI/filtered_data/             ─ Dataset filtrado (1%, ~1060 imágenes, 108 pacientes)
    └── TCGA_CS_4941_19960909/
        ├── *_1.tif                ─ Imagen RGB (R=T1, G=T1c, B=FLAIR)
        └── *_1_mask.tif           ─ Máscara binaria
```

## CRF Refinement

Pure numpy/OpenCV CRF (Krähenbühl & Koltun 2012). Gaussian + bilateral kernels in log-space. Falls back to Gaussian-only if OpenCV unavailable. For pydensecrf (Python ≤3.11): `pip install pydensecrf`.

## Tests

```bash
python -m pytest tests/ -v    # 53 passed, 2 skipped (pydensecrf)
```

## Requirements

- Python ≥3.10, PyTorch ≥2.5, CUDA 12.4+

## Citation

```bibtex
@mastersthesis{GonzalezSalas2024,
  author  = {Fernando González Salas},
  title   = {Medical Image Segmentation with Uncertainty Estimation},
  school  = {Universidade da Coruña},
  year    = {2024}
}
```
