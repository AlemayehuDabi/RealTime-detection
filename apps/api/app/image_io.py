import io
import logging

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

MAX_LONG_EDGE = 1280


class ImageDecodeError(ValueError):
    pass


def decode_and_prepare(image_bytes: bytes) -> tuple[Image.Image, bool]:
    if not image_bytes:
        raise ImageDecodeError("Empty upload")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise ImageDecodeError("File is not a supported image") from exc

    if image.format not in {"JPEG", "PNG", "JPG"}:
        raise ImageDecodeError(f"Unsupported image type: {image.format or 'unknown'}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = False
    w, h = image.size
    long_edge = max(w, h)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized = True
        logger.info("Resized image from %sx%s to %sx%s", w, h, new_w, new_h)

    return image, resized