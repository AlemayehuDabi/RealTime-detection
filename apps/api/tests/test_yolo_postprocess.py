import numpy as np

from app.yolo_postprocess import decode_yolo_onnx


def test_decode_filters_by_confidence() -> None:
    preds = np.zeros((1, 2, 84), dtype=np.float32)
    preds[0, 0, :4] = [320, 320, 80, 80]
    preds[0, 0, 4] = 0.95
    preds[0, 1, :4] = [100, 100, 40, 40]
    preds[0, 1, 5] = 0.1
    boxes, scores, class_ids = decode_yolo_onnx(preds, conf_thresh=0.25, iou_thresh=0.45)
    assert len(scores) == 1
    assert class_ids[0] == 0