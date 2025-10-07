"""Вспомогательные функции для обработки изображений."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def simple_nms(boxes, scores, iou_threshold=0.4):
    """Простейшая реализация non-maximum suppression.

    Args:
        boxes (List[list[int]]): Список прямоугольников [x1, y1, x2, y2].
        scores (List[float]): Уверенности детекции для каждого прямоугольника.
        iou_threshold (float): Порог IoU для подавления перекрывающихся боксов.

    Returns:
        List[int]: Индексы боксов, которые следует оставить.
    """
    if not boxes:
        logger.debug("simple_nms: пустой список боксов")
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
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
    logger.debug("simple_nms: после NMS осталось %s боксов", len(keep))
    return keep


def rotate_bbox(x1, y1, x2, y2, w, h, k):
    """Повернуть прямоугольник на ``k``·90° против часовой стрелки.

    Args:
        x1, y1, x2, y2: Координаты прямоугольника в исходной системе.
        w, h: Ширина и высота исходного изображения.
        k: Количество поворотов против часовой стрелки (0-3).

    Returns:
        tuple[int, int, int, int]: Координаты прямоугольника после поворота.
    """
    k = k % 4
    coords = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    if k == 1:
        pts = [(y, w - 1 - x) for x, y in coords]
    elif k == 2:
        pts = [(w - 1 - x, h - 1 - y) for x, y in coords]
    elif k == 3:
        pts = [(h - 1 - y, x) for x, y in coords]
    else:
        pts = coords

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
