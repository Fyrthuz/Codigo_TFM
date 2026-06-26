#!/bin/bash
# Train a UNet on the LGG MRI dataset
# Usage: ./scripts/train_unet.sh [data_root] [epochs]

DATA_ROOT=${1:-"./MRI/filtered_data"}
EPOCHS=${2:-10}

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "GPU detected"
    export CUDA_VISIBLE_DEVICES=0
fi

python -m src.utils.train_unet \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --save-path "./unet_model.pth"
