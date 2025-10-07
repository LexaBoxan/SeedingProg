"""Конфигурация приложения."""

from pathlib import Path
import os

# Путь к весам YOLOv8. Можно задать через переменную окружения YOLO_WEIGHTS_PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS_PATH = Path(
    os.getenv(
        "YOLO_WEIGHTS_PATH",
        str(PROJECT_ROOT / "models" /  "bestCrop.pt")
    )
)

# Путь к весам модели классификации. Можно задать переменной YOLO_CLASSIFY_WEIGHTS_PATH
DEFAULT_CLASSIFY_WEIGHTS_PATH = Path(
    os.getenv(
        "YOLO_CLASSIFY_WEIGHTS_PATH",
        str(PROJECT_ROOT / "models" / "bestKlass.pt"),
    )
)

# Параметр поворота на 90 градусов: значение k для np.rot90
ROTATE_K = 1
