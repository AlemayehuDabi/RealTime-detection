import logging

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.detector import get_detector
from app.image_io import ImageDecodeError, decode_and_prepare

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RealTime Detection API", version="0.1.0")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    from app.config import get_detector_mode

    return {"status": "ok", "detector": get_detector_mode()}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image, resized = decode_and_prepare(raw)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detector = get_detector()
    result = detector.predict(image)
    logger.info(
        "detect done width=%s height=%s resized=%s count=%s",
        result.image_width,
        result.image_height,
        resized,
        len(result.detections),
    )
    return result.model_dump()