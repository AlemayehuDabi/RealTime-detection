import random
import time

from PIL import Image

from app.coco_labels import COCO_CLASS_NAMES
from app.schemas import Box, Detection, DetectionResult


class MockDetector:
    def predict(self, image: Image.Image) -> DetectionResult:
        started = time.perf_counter()
        w, h = image.size

        rng = random.Random(42)
        count = rng.randint(1, 3)
        detections: list[Detection] = []

        for _ in range(count):
            class_id = rng.randint(0, len(COCO_CLASS_NAMES) - 1)
            box_w = rng.uniform(0.12, 0.35) * w
            box_h = rng.uniform(0.12, 0.45) * h
            x1 = rng.uniform(0, max(1.0, w - box_w))
            y1 = rng.uniform(0, max(1.0, h - box_h))
            x2 = min(w, x1 + box_w)
            y2 = min(h, y1 + box_h)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=COCO_CLASS_NAMES[class_id],
                    confidence=round(rng.uniform(0.55, 0.97), 3),
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        inference_ms = (time.perf_counter() - started) * 1000
        time.sleep(rng.uniform(0.005, 0.02))

        return DetectionResult(
            inference_ms=round(inference_ms, 2),
            image_width=w,
            image_height=h,
            detections=detections,
        )