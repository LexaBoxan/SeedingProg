# pip install ultralytics opencv-python numpy tqdm

import os
import cv2
import math
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ================== НАСТРОЙКИ ==================
SOURCE_DIR    = r"E:\_JOB_\_Python\Seeding\Photo\Pak"  # Папка с исходными фото
DATASET_ROOT  = r"E:\_JOB_\_Python\Seeding\dataset\datasetSegV5"          # КОРЕНЬ датасета (как просил)
YOLO_WEIGHTS  = r"E:\_JOB_\_Python\Seeding\models\bestCorp.pt"  # веса детектора боксов (класс Seeding)
YOLO_CLASS    = "Seeding"   # имя класса в детекторе
CONF_THRESH   = 0.25
IOU_THRESH    = 0.4         # порог IoU для NMS
IMG_SIZE      = 736         # размер инференса детектора
ROTATE_K      = 1           # если crop шире высоты — поворачиваем на 90° (np.rot90 k=1)
SPLIT_RATIO   = (0.7, 0.2, 0.1)  # 70/20/10
MIN_MASK_AREA = 50          # минимальная площадь компоненты для сохранения полигона
RNG_SEED      = 42          # воспроизводимость

# ================== ПОДГОТОВКА ПАПОК ==================
random.seed(RNG_SEED)
splits = ["train", "valid", "test"]
for sp in splits:
    for sub in ["images", "labels", "masks", "preview"]:
        Path(DATASET_ROOT, sp, sub).mkdir(parents=True, exist_ok=True)

# ================== УТИЛИТЫ ==================
def iou_xyxy(a, b):
    """IoU двух боксов [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def simple_nms(boxes, scores, iou_thr=0.5):
    """Простая NMS: возвращает индексы оставшихся боксов."""
    if not boxes:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        suppress = []
        for j in rest:
            if iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                suppress.append(j)
        idxs = np.array([j for j in rest if j not in suppress], dtype=int)
    return keep

def kmeans_segmentation(image_bgr, k=3):
    """
    Цветовая кластеризация для выделения растения.
    Возвращает бинарную маску (uint8 0/255).
    """
    # в RGB для k-means по цвету
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    Z = image_rgb.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    centers = centers.astype(np.uint8)

    segmented = centers[labels].reshape(H, W, 3)

    # берём кластер с минимальной средней яркостью (часто растение темнее фона)
    cluster_idx = int(np.argmin(centers.mean(axis=1)))
    plant_color = centers[cluster_idx]
    lower = np.clip(plant_color - 30, 0, 255)
    upper = np.clip(plant_color + 30, 0, 255)

    mask = cv2.inRange(segmented, lower, upper)

    # легкая очистка
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def save_yolo_seg_from_mask(mask, label_path, class_id=0, min_area=50):
    """
    Сохраняет полигоны маски в формате YOLO-seg (одна строка на один объект).
    Координаты нормализуются относительно размеров текущего CROP'а.
    """
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        c = c.reshape(-1, 2)
        # (опц.) можно упростить контур: c = cv2.approxPolyDP(c, epsilon=1.0, closed=True).reshape(-1,2)
        pts = []
        for (px, py) in c:
            pts.append(f"{px / w:.6f}")
            pts.append(f"{py / h:.6f}")
        lines.append(f"{class_id} " + " ".join(pts))
    if lines:
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return True
    return False

def clamp_box(x1, y1, x2, y2, W, H):
    """Обрезаем бокс границами изображения."""
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

# ================== ЗАГРУЗКА ДЕТЕКТОРА ==================
model = YOLO(YOLO_WEIGHTS)

# ================== СПИСОК ФОТО И РАЗБИЕНИЕ ПО ИСТОЧНИКАМ ==================
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
images = [f for f in os.listdir(SOURCE_DIR) if Path(f).suffix.lower() in img_exts]
images.sort()
random.shuffle(images)

n_total = len(images)
n_train = int(SPLIT_RATIO[0] * n_total)
n_valid = int(SPLIT_RATIO[1] * n_total)
src_to_split = {}
for i, name in enumerate(images):
    if i < n_train:
        sp = "train"
    elif i < n_train + n_valid:
        sp = "valid"
    else:
        sp = "test"
    src_to_split[name] = sp

# ================== ОСНОВНОЙ ЦИКЛ ==================
sample_counters = {"train": 0, "valid": 0, "test": 0}
total_kept = 0

for img_name in tqdm(images, desc="Обработка изображений"):
    img_path = str(Path(SOURCE_DIR) / img_name)
    img = cv2.imread(img_path)
    if img is None:
        tqdm.write(f"⚠️ Не удалось открыть: {img_name}")
        continue
    H, W = img.shape[:2]
    split = src_to_split[img_name]

    # --- детекция боксов ---
    results = model.predict(
        source=img,
        conf=CONF_THRESH,
        imgsz=IMG_SIZE,
        device=0,          # GPU 0; поставь 'cpu' при необходимости
        verbose=False
    )
    r = results[0]

    # собираем боксы по классу YOLO_CLASS
    boxes, scores = [], []
    for b in r.boxes:
        cls_id = int(b.cls)
        cls_name = r.names.get(cls_id, str(cls_id))
        if cls_name != YOLO_CLASS:
            continue
        conf = float(b.conf)
        x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
        cb = clamp_box(x1, y1, x2, y2, W, H)
        if cb is None:
            continue
        boxes.append(list(cb))
        scores.append(conf)

    if not boxes:
        tqdm.write(f"{img_name}: объектов класса '{YOLO_CLASS}' не найдено")
        continue

    keep_idx = simple_nms(boxes, scores, iou_thr=IOU_THRESH)

    # --- по каждому оставшемуся боксу создаём отдельный образец ---
    for j_out, j in enumerate(keep_idx):
        x1, y1, x2, y2 = boxes[j]
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        # поворот для вертикальной ориентации (если нужно)
        if crop.shape[1] > crop.shape[0]:
            crop = np.rot90(crop, k=ROTATE_K)

        # сегментация маски по цвету
        mask = kmeans_segmentation(crop, k=3)

        # если совсем пусто — пропускаем
        if cv2.countNonZero(mask) < MIN_MASK_AREA:
            continue

        # имена файлов для образца
        idx = sample_counters[split]
        base = f"{Path(img_name).stem}_s{idx:03d}"  # уникально внутри split
        sample_counters[split] += 1

        # пути сохранения
        out_img = Path(DATASET_ROOT, split, "images", f"{base}.jpg")
        out_msk = Path(DATASET_ROOT, split, "masks",  f"{base}_mask.png")
        out_lbl = Path(DATASET_ROOT, split, "labels", f"{base}.txt")
        out_prv = Path(DATASET_ROOT, split, "preview", f"{base}_preview.png")

        # сохраняем изображение и маску
        cv2.imwrite(str(out_img), crop)
        cv2.imwrite(str(out_msk), mask)

        # генерим YOLO-seg разметку (полигоны в координатах crop'а)
        ok = save_yolo_seg_from_mask(mask, str(out_lbl), class_id=0, min_area=MIN_MASK_AREA)

        # делаем превью: наложение и контуры (удобно проверять датасет)
        overlay = crop.copy()
        color_mask = np.zeros_like(crop)
        color_mask[mask > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(crop, 1.0, color_mask, 0.4, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 1)
        cv2.imwrite(str(out_prv), overlay)

        if ok:
            total_kept += 1

# ================== СОЗДАЁМ dataset.yaml С АБСОЛЮТНЫМИ ПУТЯМИ ==================
yaml_text = (
    "path: \n"  # оставляем пустым, т.к. пути абсолютные ниже
    f"train: {Path(DATASET_ROOT, 'train', 'images').as_posix()}\n"
    f"val:   {Path(DATASET_ROOT, 'valid', 'images').as_posix()}\n"
    f"test:  {Path(DATASET_ROOT, 'test',  'images').as_posix()}\n\n"
    "names:\n"
    "  0: Seeding\n"
)
with open(str(Path(DATASET_ROOT) / "dataset.yaml"), "w", encoding="utf-8") as f:
    f.write("# Датасет YOLO-seg (по боксам, один сеянец = один сэмпл)\n" + yaml_text)

print(f"✅ Готово. Всего сэмплов сохранено: {total_kept}")
print("   Структура папок:", DATASET_ROOT)
print("   YAML:", str(Path(DATASET_ROOT) / "dataset.yaml"))
