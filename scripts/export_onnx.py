#!/usr/bin/env python3
"""Export trained best.pt to ONNX. Run locally after placing weights in models/source/best.pt."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "models" / "source" / "best.pt"
DEFAULT_OUT = REPO_ROOT / "models" / "yolo26n_coco128.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX")
    parser.add_argument("--weights", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=args.opset,
        dynamic=False,
    )
    out_path = Path(exported)
    if out_path.resolve() != args.out.resolve():
        args.out.write_bytes(out_path.read_bytes())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()