from pathlib import Path

_LABELS_PATH = Path(__file__).resolve().parent / "data" / "coco80.txt"


def load_coco_class_names() -> list[str]:
    text = _LABELS_PATH.read_text(encoding="utf-8").strip()
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if len(names) != 80:
        msg = f"Expected 80 COCO class names, got {len(names)}"
        raise ValueError(msg)
    return names


COCO_CLASS_NAMES: list[str] = load_coco_class_names()