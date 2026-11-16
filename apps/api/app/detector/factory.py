from functools import lru_cache

from app.config import get_detector_mode
from app.detector.base import Detector
from app.detector.mock import MockDetector


@lru_cache
def get_detector() -> Detector:
    mode = get_detector_mode()
    if mode != "mock":
        msg = f"Detector {mode!r} is not available yet; use DETECTOR=mock"
        raise RuntimeError(msg)
    return MockDetector()