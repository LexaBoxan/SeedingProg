from ultralytics import YOLO
import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    """Читает все изображения из папки и возвращает список путей."""
    image_paths = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_paths.append(os.path.join(folder, filename))
    return image_paths

# 1. Загружаем модель сегментации. В документации указано, что достаточно передать файл .pt:contentReference[oaicite:1]{index=1}.
model = YOLO(r"E:\_JOB_\_Python\Seeding\results\seeding-seg12\weights\best.pt")


# 2. Путь к папке с изображениями (замените на свою папку)
image_folder = r"E:\_JOB_\_Python\Seeding\Photo\Pak1"
image_paths = load_images_from_folder(image_folder)

# 3. Проходим по каждому изображению
for image_path in image_paths:
    print(f"Обрабатывается {image_path}")
    # Читаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить {image_path}")
        continue

    # 4. Получаем результаты модели
    # Можно задать порог уверенности (conf), по умолчанию 0.25
    results = model.predict(img, conf=0.25)

    # Создадим копию изображения для рисования
    annotated_img = img.copy()

    # 5. Обрабатываем каждую найденную область
    for result in results:
        # result.boxes содержит координаты bbox и классы, result.masks.xy – контуры масок:contentReference[oaicite:2]{index=2}.
        for mask_points, box in zip(result.masks.xy, result.boxes):
            # Переводим координаты маски в формат np.int32 для OpenCV
            points = np.int32([mask_points])
            class_id = int(box.cls[0])
            # Получаем название класса по ID
            class_name = model.model.names[class_id]
            # Цвет маски (можно назначить случайный или по номеру класса)
            color = (0, 255, 0)  # зелёный
            cv2.fillPoly(annotated_img, points, color)

            # Рисуем bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

            # Добавляем подпись с именем класса и уверенностью
            conf_score = float(box.conf[0])
            label = f"{class_name}: {conf_score:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 6. Сохраняем и/или отображаем результат
    output_path = os.path.join("../results", os.path.basename(image_path))
    os.makedirs("../results", exist_ok=True)
    cv2.imwrite(output_path, annotated_img)
    print(f"Результат сохранён в {output_path}")

    # При желании можно показать изображение:
    cv2.imshow("Segmented", annotated_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
