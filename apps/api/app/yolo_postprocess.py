import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thresh]
    return keep


def decode_yolo_onnx(
    output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultralytics-style export: (1, 4+nc, N) or (1, N, 4+nc)."""
    if output.ndim == 3:
        batch = output[0]
    else:
        batch = output
    if batch.shape[0] in {84, 85, 80, 56} and batch.shape[0] < batch.shape[1]:
        batch = batch.T

    boxes_xywh = batch[:, :4]
    class_probs = batch[:, 4:]
    class_ids = class_probs.argmax(axis=1)
    scores = class_probs[np.arange(class_probs.shape[0]), class_ids]

    mask = scores >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if boxes_xywh.size == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.array([], dtype=np.float32), np.array([], dtype=int)

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, iou_thresh)
    keep = keep[:max_det]
    return boxes_xyxy[keep], scores[keep], class_ids[keep]