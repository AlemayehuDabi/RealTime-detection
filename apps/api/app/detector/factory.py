from functools import lru_cache

from app.config import get_detector_mode
from app.detector.base import Detector
from app.detector.mock import MockDetector


@lru_cache
def get_detector() -> Detector:
    mode = get_detector_mode()
    if mode == "mock":
        return MockDetector()
    if mode == "onnx":
        from app.detector.onnx_detector import OnnxDetector

        return OnnxDetector()
    msg = f"Unknown DETECTOR mode: {mode}"
    raise RuntimeError(msg)