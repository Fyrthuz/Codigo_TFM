#!/bin/bash
# Run the 2D UNet segmentation + uncertainty pipeline
# Usage: ./scripts/run_pipeline_2d.sh [config_path] [checkpoint_path]

CONFIG=${1:-"./configs/pipeline_2d.yaml"}
CHECKPOINT=${2:-"./unet_model.pth"}

# Detect GPU
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "GPU detected: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No GPU detected, running on CPU"
fi

python -m src.pipelines.run_unet --config "$CONFIG" --checkpoint "$CHECKPOINT"
