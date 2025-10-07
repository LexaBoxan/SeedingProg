import numpy as np
from seeding.utils import simple_nms, rotate_bbox


def test_simple_nms_basic():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    scores = [0.9, 0.8, 0.7]
    keep = simple_nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]


def test_rotate_bbox_roundtrip():
    w, h = 10, 6
    box = (1, 2, 7, 5)
    for k in range(4):
        rx1, ry1, rx2, ry2 = rotate_bbox(*box, w, h, k)
        w_rot, h_rot = (h, w) if k % 2 else (w, h)
        bx1, by1, bx2, by2 = rotate_bbox(rx1, ry1, rx2, ry2, w_rot, h_rot, (-k) % 4)
        assert (bx1, by1, bx2, by2) == box
