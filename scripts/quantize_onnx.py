#!/usr/bin/env python3
"""Dynamic INT8 quantization for ONNX (ONNX Runtime). Run after export_onnx.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "models" / "yolo26n_coco128.onnx"
DEFAULT_OUT = REPO_ROOT / "models" / "yolo26n_coco128_int8.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to dynamic INT8")
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.inp.is_file():
        raise SystemExit(f"Input ONNX not found: {args.inp}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(args.inp),
        str(args.out),
        weight_type=QuantType.QUInt8,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()