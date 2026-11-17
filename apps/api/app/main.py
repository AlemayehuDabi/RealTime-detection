import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_cors_origin_list, get_detector_mode
from app.detector import get_detector
from app.image_io import ImageDecodeError, decode_and_prepare
from app.logging_utils import setup_logging
from app.middleware import RequestContextMiddleware

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="RealTime Detection API", version="0.1.0")
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "detector": get_detector_mode()}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

    t0 = time.perf_counter()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        t_decode = time.perf_counter()
        image, resized = decode_and_prepare(raw)
        decode_ms = (time.perf_counter() - t_decode) * 1000
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detector = get_detector()
    t_infer = time.perf_counter()
    result = detector.predict(image)
    infer_ms = (time.perf_counter() - t_infer) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "detect decode_ms=%.2f infer_ms=%.2f total_ms=%.2f resized=%s count=%s size=%sx%s",
        decode_ms,
        infer_ms,
        total_ms,
        resized,
        len(result.detections),
        result.image_width,
        result.image_height,
    )
    return result.model_dump()