import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["detector"] == "mock"


def test_detect_jpeg() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buf, format="JPEG")
    buf.seek(0)
    res = client.post(
        "/detect",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["image_width"] == 64
    assert "detections" in data


def test_detect_rejects_empty() -> None:
    res = client.post(
        "/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert res.status_code == 400