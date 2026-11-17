import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.coco_labels import COCO_CLASS_NAMES
from app.config import get_settings, repo_root
from app.schemas import Box, Detection, DetectionResult
from app.yolo_image import letterbox, scale_boxes_to_original
from app.yolo_postprocess import decode_yolo_onnx


class OnnxDetector:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.onnx_model_path:
            msg = "ONNX_MODEL_PATH is not set"
            raise RuntimeError(msg)

        path = Path(settings.onnx_model_path)
        if not path.is_absolute():
            path = Path(repo_root()) / path
        if not path.is_file():
            msg = f"ONNX model not found: {path}"
            raise FileNotFoundError(msg)

        self._conf = settings.conf_thresh
        self._iou = settings.iou_thresh
        self._img_size = settings.img_size

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
        self._session = ort.InferenceSession(
            str(path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, image: Image.Image) -> DetectionResult:
        started = time.perf_counter()
        orig_w, orig_h = image.size

        arr, scale, pad = letterbox(image, self._img_size)
        blob = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        outputs = self._session.run(None, {self._input_name: blob})
        boxes, scores, class_ids = decode_yolo_onnx(
            outputs[0],
            self._conf,
            self._iou,
        )
        boxes = scale_boxes_to_original(boxes, scale, pad, orig_w, orig_h)

        detections: list[Detection] = []
        for i in range(len(scores)):
            cid = int(class_ids[i])
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=COCO_CLASS_NAMES[cid] if cid < len(COCO_CLASS_NAMES) else str(cid),
                    confidence=round(float(scores[i]), 4),
                    box=Box(
                        x1=float(boxes[i, 0]),
                        y1=float(boxes[i, 1]),
                        x2=float(boxes[i, 2]),
                        y2=float(boxes[i, 3]),
                    ),
                )
            )

        inference_ms = (time.perf_counter() - started) * 1000
        return DetectionResult(
            inference_ms=round(inference_ms, 2),
            image_width=orig_w,
            image_height=orig_h,
            detections=detections,
        )