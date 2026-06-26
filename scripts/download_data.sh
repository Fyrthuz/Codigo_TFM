#!/bin/bash
# Download datasets for the segmentation pipelines.
#
# Usage:
#   ./scripts/download_data.sh lgg                      # LGG MRI (Kaggle)
#   ./scripts/download_data.sh carvana                  # Carvana (Kaggle)
#   ./scripts/download_data.sh all                      # Everything
#
# Options:
#   --force   Re-download even if exists

set -e

DATASET=${1:-"lgg"}
shift 2>/dev/null || true

case "$DATASET" in
    lgg|carvana|all)
        python -m src.utils.download_datasets --dataset "$DATASET" "$@"
        ;;
    *)
        echo "Usage: $0 {lgg|carvana|all} [--force]"
        exit 1
        ;;
esac
