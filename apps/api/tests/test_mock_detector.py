from PIL import Image

from app.detector.mock import MockDetector


def test_mock_detector_returns_valid_result() -> None:
    image = Image.new("RGB", (320, 240))
    result = MockDetector().predict(image)
    assert result.image_width == 320
    assert result.image_height == 240
    assert 1 <= len(result.detections) <= 3
    for det in result.detections:
        assert 0 <= det.class_id < 80
        assert det.class_name
        assert 0.0 <= det.confidence <= 1.0