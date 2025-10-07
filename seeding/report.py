"""Утилиты для формирования PDF-отчёта."""

from __future__ import annotations

import io

from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from seeding.models.data_models import ObjectImage, OriginalImage
from seeding.utils import rotate_bbox


def _np_to_pil(img: np.ndarray) -> Image.Image:
    """Преобразует изображение NumPy (BGR или оттенки серого) в `PIL.Image`."""
    if img is None:
        raise ValueError("Image is None")
    if img.ndim == 3:
        if img.shape[2] == 3:
            # convert BGR to RGB
            return Image.fromarray(img[:, :, ::-1])
        else:
            return Image.fromarray(img)
    else:
        return Image.fromarray(img)


def _annotate_image(img: np.ndarray, objects: list[ObjectImage]) -> np.ndarray:
    """Наносит на изображение рамки объектов с порядковыми номерами."""
    annotated = img.copy()
    for i, obj in enumerate(objects, start=1):
        if obj.bbox:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                str(i),
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        if obj.image_all_class and obj.bbox:
            k = getattr(obj, "rotation_k", 0) % 4
            h_rot, w_rot = obj.image[0].shape[:2] if obj.image else (0, 0)
            for cls in obj.image_all_class:
                if cls.bbox:
                    lx1, ly1, lx2, ly2 = cls.bbox
                    if k and h_rot and w_rot:
                        ux1, uy1, ux2, uy2 = rotate_bbox(
                            lx1, ly1, lx2, ly2, w_rot, h_rot, (-k) % 4
                        )
                    else:
                        ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2
                    x1 = obj.bbox[0] + ux1
                    y1 = obj.bbox[1] + uy1
                    x2 = obj.bbox[0] + ux2
                    y2 = obj.bbox[1] + uy2
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return annotated


def _pil_to_buf(image: Image.Image, *, quality: int = 70) -> io.BytesIO:
    """Сохраняет изображение в буфер JPEG для вставки в PDF.

    Args:
        image: Изображение PIL.
        quality: Качество JPEG (1-95), чем ниже — тем меньше размер.

    Returns:
        Буфер с JPEG-данными.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer


def _object_to_pil(obj: ObjectImage) -> Image.Image | None:
    """Преобразует объект в `PIL.Image`, если возможно."""
    if not obj.image:
        return None
    img = obj.image[0]
    if isinstance(img, np.ndarray):
        return _np_to_pil(img)
    if isinstance(img, Image.Image):
        return img
    return None


def _rl_image_from_pil(image: Image.Image, max_w: float, max_h: float) -> RLImage:
    """Создаёт элемент ReportLab с ограничением по размерам.

    Args:
        image: Исходное изображение PIL.
        max_w: Максимальная ширина в пунктах.
        max_h: Максимальная высота в пунктах.

    Returns:
        Элемент ``RLImage`` с подогнанными размерами.
    """
    aspect = image.height / float(image.width)
    width_pt = max_w
    height_pt = width_pt * aspect
    if height_pt > max_h:
        height_pt = max_h
        width_pt = height_pt / aspect
    return RLImage(_pil_to_buf(image), width=width_pt, height=height_pt)

def create_pdf_report(data: OriginalImage, output_path: str) -> None:
    """Создаёт PDF-отчёт с результатами детекции.

    На каждой странице показывается исходное изображение с рамками и номерами,
    таблица характеристик и уменьшенные копии каждого найденного объекта.
    Изображения сохраняются в формате JPEG с качеством 70 для снижения размера
    итогового файла.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for idx, img in enumerate(data.images):
        objs: list[ObjectImage] = []
        if data.class_object_image and len(data.class_object_image) > idx:
            objs = data.class_object_image[idx]

        annotated_img = _annotate_image(img, objs)
        pil_img = _np_to_pil(annotated_img)

        story.append(Paragraph(f"Страница {idx + 1}", styles["Heading1"]))

        max_width = 160 * mm
        max_height = 200 * mm
        img_elem = _rl_image_from_pil(pil_img, max_width, max_height)
        story.append(img_elem)
        story.append(Spacer(1, 5 * mm))

        table_data = [["#", "Class", "Confidence", "BBox"]]
        for i, obj in enumerate(objs, start=1):
            bbox = obj.bbox if obj.bbox else ("", "", "", "")
            table_data.append([
                str(i),
                obj.class_name,
                f"{obj.confidence:.2f}",
                str(bbox),
            ])

        table = Table(table_data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(table)
        story.append(PageBreak())

        obj_per_page = 3
        for i, obj in enumerate(objs, start=1):
            pil_obj = _object_to_pil(obj)
            if pil_obj is None:
                continue
            story.append(Paragraph(f"Объект {i}", styles["Heading3"]))
            crop_elem = _rl_image_from_pil(pil_obj, 60 * mm, 80 * mm)
            story.append(crop_elem)
            story.append(Spacer(1, 5 * mm))
            if i % obj_per_page == 0 and i != len(objs):
                story.append(PageBreak())

        if idx < len(data.images) - 1:
            story.append(PageBreak())

    doc.build(story)
