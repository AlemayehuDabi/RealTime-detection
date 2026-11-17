import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    detector: str = "mock"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    onnx_model_path: str = Field(
        default="",
        validation_alias="ONNX_MODEL_PATH",
    )
    conf_thresh: float = Field(default=0.25, validation_alias="CONF_THRESH")
    iou_thresh: float = Field(default=0.45, validation_alias="IOU_THRESH")
    img_size: int = Field(default=640, validation_alias="IMG_SIZE")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_detector_mode() -> str:
    return get_settings().detector.lower()


def get_cors_origin_list() -> list[str]:
    raw = get_settings().cors_origins.strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def repo_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."),
    )