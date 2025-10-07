from ultralytics import YOLO

if __name__ == "__main__":
    # Обычная модель детекции
    model = YOLO(r"E:\_JOB_\_Python\Seeding\models\yolov8s.pt")

    model.train(
        # УКАЖИТЕ ваш data.yaml для детекции:
        # (внутри должны быть пути к train/val images и к labels в формате YOLO)
        data=r"E:\_JOB_\_Python\Seeding\dataset\datasetKlassV3\data.yaml",

        project=r"E:\_JOB_\_Python\Seeding\results",
        name="seeding-klass",

        # --- Ресурсы (RTX 3050 4GB) ---
        device=0,
        imgsz=1024,     # если вылетает по памяти -> 512
        batch=8,       # если OOM -> 4
        workers=4,     # Windows обычно стабильнее с 2 (или даже 0)
        cache=True,
        amp=True,      # FP16

        # --- Оптимизация ---
        optimizer="SGD",   # для детекции классика
        lr0=0.00125,       # масштаб lr0 под batch=8 (из 0.01 при nbs=64)
        lrf=0.01,          # финальный LR = lr0 * lrf (cosine decay)
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3,

        # --- Аугментации (умеренные для стабильности) ---
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.08, scale=0.5, shear=0.0,
        mosaic=0.5,        # можно 0.3–0.5; 0.0 полностью отключает
        mixup=0.10,        # при необходимости можно 0.0
        close_mosaic=32,   # раннее отключение мозаики к концу обучения
        rect=True,         # прямоугольные батчи экономят память
        multi_scale=False, # включайте позже, если нужно

        # --- Обучение/логирование ---
        epochs=128,
        save_period=25,
        # patience=50,     # можно включить раннюю остановку
        # resume=True,     # для продолжения обучения с последней точки
    )
