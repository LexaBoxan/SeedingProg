# -*- coding: utf-8 -*-
"""
PDF -> высокое разрешение (scale), YOLO детекция, вырезка bbox и сохранение PNG.
Цвета НЕ меняем: работаем в RGB, а сохраняем через PIL (тоже RGB), чтобы не было "синих" картинок.
"""

from __future__ import annotations
from pathlib import Path
import re
import sys
from typing import Tuple, List

import numpy as np
import fitz  # PyMuPDF
from ultralytics import YOLO
from PIL import Image  # используем PIL для корректного сохранения RGB

# ===================== НАСТРОЙКИ =====================

PDF_PATH = Path(r"E:\_JOB_\_Python\Seeding\Photo\Pak3.pdf")
OUTPUT_DIR = Path(r"E:\_JOB_\_Python\Seeding\Photo\Pak1_crops")

# Ваши веса YOLO. Если оставить None — возьмём yolov8n.pt
WEIGHTS_PATH = Path(r"E:\_JOB_\_Python\Seeding\models\bestCorp.pt")

SCALE = 2.0      # во сколько раз увеличить рендер (ширину/высоту)
CONF = 0.25      # порог уверенности YOLO
IOU  = 0.40      # порог IoU для NMS
ROTATE_LANDSCAPE = True   # поворачивать crop на 90°, если ширина > высоты
FILENAME_PREFIX  = ""     # префикс к имени файлов (можно "")

# =====================================================


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def render_page_to_image_rgb(page: fitz.Page, scale: float) -> np.ndarray:
    """
    Рендер страницы PDF в RGB (HxWx3). Если пришла RGBA — просто отбрасываем альфу (канал A).
    НИКАКОЙ конвертации каналов (чтобы не испортить цвета).
    """
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)  # bytes (RGB или RGBA)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]  # RGBA -> RGB (отрезали альфу)
    # Если pix.n == 3 — уже RGB
    return img


def rotate_if_landscape(crop_rgb: np.ndarray, enabled: bool = True) -> np.ndarray:
    """
    Повернуть на 90° (np.rot90), если ширина > высоты. Работает в RGB.
    """
    if enabled and crop_rgb.shape[1] > crop_rgb.shape[0]:
        crop_rgb = np.rot90(crop_rgb)
    return crop_rgb


def simple_nms(boxes: List[List[int]], scores: List[float], iou_threshold: float) -> List[int]:
    """
    Простейший NMS (аналог вашей seeding.utils.simple_nms).
    Возвращает индексы боксов, оставшихся после подавления.
    """
    if not boxes:
        return []
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    x1, y1, x2, y2 = boxes_np.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores_np.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def detect_objects_rgb(model: YOLO, image_rgb: np.ndarray, conf: float) -> tuple[list[list[int]], list[float], list[str]]:
    """
    Детекция YOLO на RGB. Возвращает списки боксов, вероятностей, имён классов.
    """
    # Ultralytics принимает numpy-изображения; внутренние конвертации он делает сам.
    # Важно: мы сохраняем crop'ы именно из RGB-копии страницы, а не из BGR.
    results = model.predict(image_rgb, verbose=False, conf=conf)

    boxes: list[list[int]] = []
    scores: list[float] = []
    class_names: list[str] = []

    for res in results:
        if res.boxes is None:
            continue
        xyxy = res.boxes.xyxy
        confs = res.boxes.conf
        clses = res.boxes.cls
        for box, score, cls in zip(xyxy, confs, clses):
            x1, y1, x2, y2 = map(int, box.tolist())
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            name = model.names[int(cls)] if hasattr(model, "names") else str(int(cls))
            class_names.append(name)
    return boxes, scores, class_names


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    """
    Обрезать bbox по границам изображения. Гарантирует минимальный размер 1×1.
    """
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def sanitize(name: str) -> str:
    """Безопасное имя файла (убираем пробелы/спецсимволы)."""
    return re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)


def save_png_rgb(path: Path, img_rgb: np.ndarray) -> None:
    """
    Сохранение PNG через PIL в RGB, чтобы исключить путаницу каналов.
    НЕ используем cv2.imwrite, чтобы не получить «синие» картинки.
    """
    Image.fromarray(img_rgb, mode="RGB").save(str(path), format="PNG")


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    weights_path: Path | None,
    scale: float,
    conf: float,
    iou: float,
    rotate_landscape: bool,
    filename_prefix: str,
) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Не найден PDF: {pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем YOLO
    if weights_path and weights_path.exists():
        model = YOLO(str(weights_path))
        log(f"YOLO веса: {weights_path}")
    else:
        model = YOLO("yolov8n.pt")
        log("Внимание: custom веса не заданы — используем yolov8n.pt")

    # Открытие документа
    doc = fitz.open(str(pdf_path))
    log(f"Открыт PDF: {pdf_path}, страниц: {len(doc)}")

    total_crops = 0

    for page_idx, page in enumerate(doc, start=1):
        # Рендерим RGB с нужным масштабом (качество ↑, пикселей больше)
        img_rgb = render_page_to_image_rgb(page, scale=scale)
        h, w = img_rgb.shape[:2]

        # Детекция
        boxes, scores, class_names = detect_objects_rgb(model, img_rgb, conf=conf)

        # NMS
        keep = simple_nms(boxes, scores, iou_threshold=iou)
        log(f"Страница {page_idx}: детекций={len(boxes)}, после NMS={len(keep)}")

        # Вырезка и сохранение
        for obj_idx, i in enumerate(keep, start=1):
            x1, y1, x2, y2 = clamp_box(*boxes[i], w, h)
            crop_rgb = img_rgb[y1:y2, x1:x2].copy()
            crop_rgb = rotate_if_landscape(crop_rgb, enabled=rotate_landscape)

            cls  = class_names[i] if i < len(class_names) else "obj"
            conf_v = scores[i] if i < len(scores) else 0.0

            stem = pdf_path.stem  # например, Pak1
            base = f"{filename_prefix}{stem}_p{page_idx}_obj{obj_idx}_{cls}_{conf_v:.2f}"
            base = sanitize(base)
            out_path = output_dir / f"{base}.png"

            # Сохраняем ЧЕРЕЗ PIL (RGB), чтобы не «синить»
            save_png_rgb(out_path, crop_rgb)
            total_crops += 1

    log(f"Готово: сохранено {total_crops} кропов → {output_dir}")


def main() -> None:
    try:
        process_pdf(
            pdf_path=PDF_PATH,
            output_dir=OUTPUT_DIR,
            weights_path=WEIGHTS_PATH,
            scale=SCALE,
            conf=CONF,
            iou=IOU,
            rotate_landscape=ROTATE_LANDSCAPE,
            filename_prefix=FILENAME_PREFIX,
        )
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
