import os
from functools import lru_cache


@lru_cache
def get_detector_mode() -> str:
    return os.environ.get("DETECTOR", "mock").lower()