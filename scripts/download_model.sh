#!/usr/bin/env bash
# Placeholder: copy your Colab best.pt into models/source/best.pt, then run export/quantize scripts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models/source"

echo "Expected layout:"
echo "  $ROOT/models/source/best.pt"
echo ""
echo "After copying weights:"
echo "  pip install ultralytics onnx onnxruntime"
echo "  python scripts/export_onnx.py"
echo "  python scripts/quantize_onnx.py"
echo ""
echo "Then set apps/api/.env: DETECTOR=onnx ONNX_MODEL_PATH=models/yolo26n_coco128_int8.onnx"