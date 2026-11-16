from fastapi import FastAPI

from app.config import get_detector_mode

app = FastAPI(title="RealTime Detection API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "detector": get_detector_mode()}