# TFM Segmentation & Uncertainty Estimation Repository

## Table of Contents
- [Overview](#overview)
- [Folder & Script Functionality](#folder--script-functionality)
- [Installation](#installation)
- [How to Run the Main Scripts](#how-to-run-the-main-scripts)
- [Data Preparation](#data-preparation)
- [Extracting Data](#extracting-data)
- [MRI Dataset Preparation and Usage](#mri-dataset-preparation-and-usage)
- [Running and Evaluating UniVerSeg, MedSAM, and UNet on MRI](#running-and-evaluating-universeg-medsam-and-unet-on-mri)
- [Bash Scripts for Pipeline Execution](#bash-scripts-for-pipeline-execution)
- [Citation](#citation)

---

## Overview
This repository contains code for training, evaluating, and analyzing deep learning models (mainly UNet and DenseNN) for image segmentation and uncertainty estimation using Monte Carlo Dropout and other advanced techniques. It includes utilities for dataset preparation, model training, evaluation, and visualization.

## Folder & Script Functionality

### Main Folders
- `models/`         : Contains model definitions (e.g., UNet, DenseNN).
- `utils/`          : Utility scripts for data loading, augmentation, and metrics.
- `data/`           : Additional datasets and test images (including MNIST and Carvana if used).
- `results/`        : Output results, figures, and evaluation metrics.
- `images/`         : Example images and figures for documentation.
- `MRI/`            : MRI datasets and related files.

### Key Scripts
- `train.py`        : Trains a DenseNN model on MNIST (example classification pipeline).
- `unet.py`         : UNet model definition and training for segmentation (MRI dataset by default).
- `test_unet.py`    : Runs inference and uncertainty estimation on a test image using a trained UNet.
- `mc_dropout.py`   : Implements Monte Carlo Dropout wrapper for uncertainty estimation.
- `scaled_mc_dropout_cross_entropy.py`, `scaled_mc_dropout_hamming.py`, etc.: Scaled MC Dropout methods (see Carvana section below).
- `compare_scaled_methods.py` : Compares different scaled MC Dropout methods (mainly for Carvana, not MRI metrics).
- `utils/dataset_mri.py`/`dataset_mri_combined.py`: Dataset loaders for MRI datasets.
- `utils/general.py`, `utils/utils.py`: Utility functions for metrics, augmentation, and preprocessing.

---

## Datasets Overview

### Carvana Dataset (for Scaled MC Dropout Methods)
The Carvana dataset is used for running and testing the scaled MC Dropout methods (e.g., `scaled_mc_dropout_cross_entropy.py`, `scaled_mc_dropout_hamming.py`, and `compare_scaled_methods.py`).

- **Preparation**: You must download the Carvana dataset (images and RLE masks) from the official source.
- **Usage**: The script `prepare_carvana_dataset.py` (if present) can be used to convert RLE masks to image masks. Place the images and masks in a `carvana/` folder as required by the scripts.
- **Note**: The Carvana dataset is not used for extracting final metrics, but for checking if the scaled MC Dropout methods can be useful in the MRI dataset context. The script `compare_scaled_methods.py` is also focused on Carvana and is not used for MRI metrics, there is also a version using the MRI dataset `compare_scaled_methods_mri.py`.

---

### MRI Dataset (for Metrics Extraction)
The MRI dataset is used for extracting and comparing metrics (IoU, Dice, ECE, etc.) for segmentation models and uncertainty estimation methods.

- **Preparation**: Place your MRI data in the `MRI/` folder, organized as described in the MRI section below.
- **Usage**: All metric extraction and comparison scripts (e.g., `compare_scaled_metrics_mri.py`, `combined_pipeline_*.py`) use the MRI dataset.

> **Note:** In the final experiments and results, only the MRI dataset is used for metrics and comparisons. Carvana and the scripts using it (including `compare_scaled_methods.py`) are for demonstration and method development only.

---

## Installation

1. **Clone the repository**
   ```cmd
   git clone https://github.com/Fyrthuz/Codigo_TFM
   cd Codigo
   ```

2. **Install dependencies**
   It is recommended to use a virtual environment:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   - torch, torchvision, tensorflow, keras, numpy, matplotlib, scikit-learn, pandas, tqdm, scikit-image, imageio, medpy, opencv-python, seaborn, pyyaml
   - External: [segment-anything](https://github.com/facebookresearch/segment-anything), [UniverSeg](https://github.com/JJGO/UniverSeg)

## How to Run the Main Scripts

- **Train DenseNN on MNIST**
  ```cmd
  python train.py
  ```
  Model will be saved as `mnist_model.pth`.

- **Train UNet on MRI dataset**
  Edit dataset path in `unet.py` if needed, then run:
  ```cmd
  python unet.py
  ```
  Model will be saved as `unet_model.pth`.

- **Test UNet and visualize uncertainty**
  ```cmd
  python test_unet.py
  ```
  Shows segmentation and uncertainty maps for a test image.

- **Monte Carlo Dropout/Uncertainty Calibration**
  ```cmd
  python scaled_mc_dropout_cross_entropy.py
  ```
  Runs MC Dropout calibration and uncertainty scaling on segmentation models.

## Data Preparation
- **MRI**: Place MRI data in `MRI/` and update paths in scripts as needed.
- **MNIST**: Downloaded automatically by `train.py`.

## Extracting Data
- For MRI datasets, unzip as needed and update paths in scripts.

## MRI Dataset Preparation and Usage

### 1. Dataset Structure & Preparation
- The MRI dataset should be organized as follows:
  ```
  MRI/
    kaggle_3m/
      <case_folder>/
        <image>.tif
        <image>_mask.tif
      ...
    filtered_data/
      <case_folder>/
        <image>.tif
        <image>_mask.tif
      ...
  ```
- Each `<case_folder>` contains paired images and masks. Masks must have the same name as the image, with `_mask` appended before the extension.
- To filter out slices with empty masks, use:
  ```cmd
  python utils\filter_data_mri.py
  ```
  This will create a `filtered_data/` directory with only valid image/mask pairs.

### 2. Loading the MRI Dataset
- The dataset loader is in `utils/dataset_mri_combined.py` (`LGGSegmentationDataset`).
- Example usage:
  ```python
  from utils.dataset_mri_combined import LGGSegmentationDataset
  dataset = LGGSegmentationDataset('./MRI/filtered_data')
  ```

### 3. Training UNet on MRI
- Edit the dataset path in `unet.py` if needed.
- Run:
  ```cmd
  python unet.py
  ```
  This will train and save a UNet model on the MRI dataset.

---

## Running and Evaluating UniVerSeg, MedSAM, and UNet on MRI

### 1. UniVerSeg
- Install UniVerSeg:
  ```cmd
  pip install git+https://github.com/JJGO/UniverSeg.git
  ```
- Use `combined_pipeline_universeg.py` for inference and metrics:
  ```cmd
  python combined_pipeline_universeg.py --config combined_config_universeg.yaml
  ```
  - Edit `combined_config_universeg.yaml` to set the correct MRI data path.
  - The script will output segmentation results and metrics (IoU, Dice, ECE, etc.) for UniVerSeg on the MRI dataset.

### 2. MedSAM
- Install MedSAM and download the checkpoint as described in the script header.
- Use `combined_pipeline_medsam.py`:
  ```cmd
  python combined_pipeline_medsam.py --config combined_config_medsam.yaml
  ```
  - Edit `combined_config_medsam.yaml` to set the correct MRI data path and MedSAM checkpoint.
  - The script will output segmentation and metrics for MedSAM.

### 3. UNet (Own Model)
- Use `combined_pipeline_own_model.py`:
  ```cmd
  python combined_pipeline_own_model.py --config combined_config.yaml
  ```
  - Edit `combined_config.yaml` to set the MRI data path.
  - This script will run inference and compute metrics for your trained UNet.

---

## Bash Scripts for Pipeline Execution

Several `.sh` scripts are provided for running the main pipelines (useful for batch or cluster execution):
- `execute_gpu_medsam.sh`         : Runs the MedSAM pipeline on GPU.
- `execute_gpu_universeg.sh`      : Runs the UniVerSeg pipeline on GPU.
- `execute_gpu_universeg_mean.sh` : Runs UniVerSeg (mean variant) on GPU.
- `execute_gpu_universeg_unique.sh`: Runs UniVerSeg (unique channel variant) on GPU.
- `execution_pipeline.sh`         : General pipeline execution script.

> **Note:** On Windows, you can inspect these scripts for the commands and run the equivalent Python commands in your terminal, or use Windows Subsystem for Linux (WSL) to execute them directly.

## Citation
If you use this code, please cite the original authors and repositories referenced in the requirements.
