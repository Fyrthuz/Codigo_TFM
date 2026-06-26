#!/bin/bash
# Run UniVerSeg few-shot pipeline
# Usage: ./scripts/run_foundation.sh

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
    export CUDA_VISIBLE_DEVICES=0
fi

python -m src.pipelines.run_foundation --config ./configs/foundation_universeg.yaml
