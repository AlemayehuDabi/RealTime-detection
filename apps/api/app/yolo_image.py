import numpy as np
from PIL import Image


def letterbox(
    image: Image.Image,
    new_size: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with aspect ratio, pad to square new_size. Returns CHW-ready RGB array [H,W,3]."""
    w, h = image.size
    scale = min(new_size / w, new_size / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (new_size, new_size), color)
    pad_x = (new_size - nw) / 2
    pad_y = (new_size - nh) / 2
    canvas.paste(resized, (int(pad_x), int(pad_y)))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr, scale, (pad_x, pad_y)


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    pad_x, pad_y = pad
    out = boxes_xyxy.copy()
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_h)
    return out