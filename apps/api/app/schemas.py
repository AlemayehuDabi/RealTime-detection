from pydantic import BaseModel, Field


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: Box


class DetectionResult(BaseModel):
    inference_ms: float
    image_width: int
    image_height: int
    detections: list[Detection]