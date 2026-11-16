from typing import Protocol

from PIL import Image

from app.schemas import DetectionResult


class Detector(Protocol):
    def predict(self, image: Image.Image) -> DetectionResult: ...