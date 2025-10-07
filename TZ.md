# Seeding — Архитектура v2 (с учётом UI)

Ниже — полностью пересобранная структура проекта и архитектурная схема с учётом графического интерфейса на PyQt + qt-material и каскадного пайплайна детекции/измерений.

---

## 1. Цели архитектуры

- Разделить **UI** и **ядро** (детекция/обработка), чтобы их можно было развивать независимо.
- Обеспечить **чистый сервисный слой** между интерфейсом и вычислительным кодом.
- Поддержать **CLI** и **GUI** поверх одного и того же API.
- Минимизировать состояние, хранить результаты в прозрачной файловой структуре (JSON/CSV/изображения), БД — опциональна.

### Обновление

Метод `find_all_seedlings` в интерфейсе выполняет детекцию сеянцев
последовательно в основном потоке. Это избавляет приложение от крашей,
которые возникали при одновременном запуске нескольких `QThread` для
каждой страницы.

---

## 2. Итоговая структура каталогов

```
project/
├─ app/                          # Графический интерфейс (PyQt)
│  ├─ main.py                    # вход в GUI-приложение
│  ├─ widgets/                   # кастомные виджеты
│  │  ├─ viewer.py               # QGraphicsView с зумом, оверлеями
│  │  ├─ left_tools.py           # левая панель инструментов
│  │  ├─ right_explorer.py       # проводник файлов
│  │  └─ dialogs.py              # диалоги (настройки, прогресс, выбор шага)
│  ├─ controllers/               # слой координации UI↔сервис
│  │  └─ app_controller.py       # связывает сигналы UI с CoreService
│  ├─ themes/                    # темы qt-material (необязательно)
│  └─ resources/                 # иконки и т.п.
│
├─ core/                         # Ядро обработки (чистая логика)
│  ├─ __init__.py
│  ├─ config.py                  # загрузка настроек, пути по умолчанию
│  ├─ types.py                   # dataclass'ы: Scale, детекции, измерения
│  ├─ io_paths.py                # правила именования и расположения результатов
│  ├─ scale.py                   # калибровка (расчёт mm/px, сохранение/загрузка)
│  ├─ detectors/
│  │  ├─ seedling.py            # YOLO det#1 — bbox по сеянцам
│  │  └─ parts.py               # YOLO det#2 — bbox частей на кропах
│  ├─ vision/
│  │  ├─ crops.py               # вырезки с паддингом
│  │  ├─ binarize.py            # порог/морфология
│  │  ├─ skeleton.py            # скелет, главная ось
│  │  └─ measure.py             # длины/углы/ширины, px→мм
│  ├─ visualize.py              # отрисовка боксов/скелетов/подписей
│  ├─ export.py                 # JSON/CSV/Excel сохранение
│  └─ service.py                # CoreService — единая точка входа для GUI/CLI
│
├─ cli/
│  └─ saplings_cli.py           # CLI: calibrate / process / batch
│
├─ data/
│  ├─ seedling/                 # датасет det#1
│  └─ parts/                    # датасет det#2 (кропы)
│
├─ models/                      # веса YOLO (det#1/ det#2)
├─ results/                     # выходы обработки
├─ config/
│  └─ settings.yaml             # глобальные настройки (пути, пороги, цвета)
└─ requirements.txt
```

> Примечание: текущий файл «Saplings UI — PyQt минимальный каркас (qt-material)» разбиваем на `app/main.py`, `widgets/*`. Существующий код легко перенести по местам (см. §5.1).

---

## 3. Главное разделение по слоям

### 3.1. UI (слой представления)

- **Задача**: показывать изображения и оверлеи, ловить клики, запускать операции, показывать прогресс/ошибки.
- **Не содержит** алгоритмов детекции/измерения.

### 3.2. Controllers (связующее звено)

- Преобразуют сигналы UI в вызовы CoreService.
- Управляют асинхронным запуском долгих задач (через `QThread`/`QRunnable`).

### 3.3. Core (доменная логика)

- Пакет с модульной логикой: детекция, кропы, бинаризация, скелет, меры, экспорт.
- Один **CoreService** — фасад для GUI/CLI.

---

## 4. Поток данных (Data Flow)

1. Пользователь открывает изображение → UI отображает его.
2. Калибровка: UI собирает 2 клика → `CoreService.set_scale(image_path, step_mm, p1, p2)` → `scale.json` рядом с файлом.
3. Обработка: `CoreService.process(image_path, model_seedling, model_parts, options)`
   - detect#1 на полном кадре → список боксов сеянцев.
   - crop каждого сеянца → detect#2 (части).
   - для `stem`/`root`: бинаризация→скелет→главная ось→длина, угол.
   - для `inflorescence`: ширина (по границам/MinAreaRect).
   - px→мм по `scale.json`.
   - overlay.jpg, instances.json, measurements.csv (+xlsx — опционально).
4. UI получает результат (пути файлов/структуры) и рисует оверлей поверх изображения (без повторного расчёта).

---

## 5. Интерфейсы и ключевые компоненты

### 5.1. UI: основные файлы

- `app/main.py` — создаёт `QApplication`, применяет qt-material, инициализирует `AppController` и главное окно.
- `app/widgets/viewer.py` — виджет просмотра (на базе текущего `ImageViewer`):
  - методы: `load_image(path)`, `set_overlay(polylines, boxes)`, режим калибровки (2 клика → сигнал).
- `app/widgets/left_tools.py` — кнопки: **Открыть**, **Калибровка**, **Detect**, **Process**, **Сохранить overlay**.
- `app/widgets/right_explorer.py` — проводник на основе `QFileSystemModel`.
- `app/controllers/app_controller.py` — подписывается на сигналы виджетов, вызывает `CoreService`.

### 5.2. CoreService (core/service.py)

Фасад для всех операций. Примерный интерфейс:

```python
class CoreService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.det_seedling = SeedlingDetector(settings.det1_path, conf=settings.det_conf)
        self.det_parts = PartsDetector(settings.det2_path, conf=settings.det_conf, iou=settings.det_iou)

    # калибровка
    def set_scale(self, image_path: Path, p1: tuple[float,float], p2: tuple[float,float], step_mm: float) -> Scale:
        scale = compute_scale_from_points(p1, p2, step_mm)
        save_scale_near_image(image_path, scale)
        return scale

    # единый вход обработки
    def process(self, image_path: Path, out_dir: Path | None = None) -> dict:
        # 1) загрузка scale.json
        scale = load_scale_near_image(image_path)
        # 2) детекция
        seed_boxes = self.det_seedling.predict(read_bgr(image_path))
        # 3) цикл по сеянцам: кроп → детекция частей → измерения
        # 4) визуализация и экспорт в out_dir
        # 5) вернуть словарь с путями файлов и агрегированными метриками
        ...
```

### 5.3. Типы данных (core/types.py)

- `Scale { mm_per_px: float, H: np.ndarray | None }`
- `BBox { x:int, y:int, w:int, h:int, score:float }`
- `PartDetection { cls:str, bbox:BBox, score:float }`
- `Polyline = list[tuple[int,int]]`
- `PartMeasure { length_px: float, angle_deg: float | None, polyline: Polyline, width_px: float }`
- `SeedlingResult { seed_bbox:BBox, parts: dict[str, PartMeasure] }`

### 5.4. Настройки (core/config.py, config/settings.yaml)

- Пути по умолчанию: models, results.
- Порог `conf`, `iou`, `imgsz` для YOLO.
- Визуальные цвета, толщина линий для overlay.
- Флаги: сохранять Excel, имя листа, локаль для CSV.

---

## 6. Асинхронность и отклики UI

- Долгие операции (детекции, целая «process») запускать в **QThread** или через **QRunnable/QThreadPool**.
- Контроллер публикует сигналы прогресса: «загрузка модели», «детекция N/M», «сохранение результатов».
- UI блокируется только мягко: кнопки отключены, но окно отвечает (индикатор в статус-баре).

---

## 7. Результаты и файловая схема

```
results/
  <image_stem>/
    scale.json                # {"mm_per_px": ...}
    overlay.jpg               # отрисовка детекций/скелетов/подписей
    instances.json            # seedling bbox + parts + полилинии + скор
    measurements.csv          # таблица по сеянцам (мм)
    measurements.xlsx         # опционально
```

Формат `instances.json` (пример):

```json
{
  "image": "IMG_0001.jpg",
  "mm_per_px": 0.0213,
  "seedlings": [
    {
      "id": 1,
      "seed_bbox": [x,y,w,h],
      "parts": {
        "stem": {"bbox": [..], "length_px": 310.5, "angle_deg": 84.1, "polyline": [[x,y],... ]},
        "root": {"bbox": [..], "length_px": 250.2, "angle_deg": 95.6, "polyline": [[x,y],... ]},
        "inflorescence": {"bbox": [..], "width_px": 28.4}
      }
    }
  ]
}
```

---

## 8. CLI (совместимый с CoreService)

`cli/saplings_cli.py` вызывает те же методы `CoreService`:

```
python -m cli.saplings_cli calibrate path/to/img.jpg --step-mm 5
python -m cli.saplings_cli process path/to/img_or_dir --export-excel
```

Опции (сокращённо): `--models det1.pt det2.pt`, `--conf 0.25`, `--iou 0.45`, `--imgsz 1280/640`.

---

## 9. Нужна ли БД?

- На старте — **нет**. Данные прозрачно лежат в файловой структуре; их легко проверять и версионировать.
- Если потребуется каталог больших партий и быстрые фильтры/поиск — добавить **SQLite** (таблицы `images`, `seedlings`, `measures`), и сделать опциональный экспорт в БД.

---

## 10. Интеграция существующего кода

- Текущий файл UI (из канваса) разбить на:
  - `app/widgets/viewer.py` — класс `ImageViewer` как есть + метод `draw_overlay(items)` для временных отрисовок.
  - `app/widgets/left_tools.py`, `app/widgets/right_explorer.py` — без логики обработки.
  - `app/controllers/app_controller.py` — перенести «слоты» `on_open`, `on_calibrate`, `on_process`, подключить `CoreService`.
- Модули ядра перенести в `core/` (из прежнего `src/`), переименовать `export_io.py` → `export.py`.

---

## 11. Дорожная карта внедрения

1. **Рефактор UI**: разбить на `app/*`, добавить контроллер, не трогая функциональность.
2. **Вынести CoreService** и перенести модули в `core/*` с минимальными правками импортов.
3. **Связать** кнопку **Process** с `CoreService.process()` и отрисовать overlay в центре.
4. **Добавить прогресс** (QThreadPool + сигналы), возможность остановки операции.
5. **Параметры** в `settings.yaml` + диалог «Настройки».
6. (Опционально) **SQLite** и вкладка «Сводка партии».

---

## 12. Мини-контракты между слоями

- UI никак не зависит от ultralytics/opencv напрямую — только через `CoreService`.
- `CoreService.process()` гарантированно возвращает:

```python
{
  "out_dir": Path,
  "overlay_path": Path,
  "csv_path": Path,
  "json_path": Path,
  "stats": {"seedlings": N, "errors": K}
}
```

- Все исключения внутри ядра поднимаются как `CoreError` с человекочитаемым сообщением; UI показывает диалог ошибки.

---

## 13. Настройки по умолчанию (config/settings.yaml)

```yaml
models:
  det1: models/seedling.pt
  det2: models/parts.pt

yolo:
  conf: 0.25
  iou: 0.45
  imgsz_seedling: 1280
  imgsz_parts: 640

results_dir: results
export:
  excel: true

visual:
  colors:
    seedling: [0, 180, 255]
    stem: [255, 160, 0]
    root: [180, 0, 255]
    inflorescence: [0, 200, 0]
  thickness: 2
```

---

## 14. Пример использования в коде контроллера

```python
# внутри AppController
self.core = CoreService(load_settings())

# калибровка
scale = self.core.set_scale(img_path, (x1,y1), (x2,y2), step_mm)

# обработка
res = self.core.process(img_path)  # вернёт пути файлов и цифры
self.viewer.show_overlay_from_file(res["overlay_path"])  # или draw_overlay(...)
```

---

Эта схема закрывает ваши требования:

- слева инструменты, справа проводник, по центру изображение;
- один сервис для GUI/CLI;
- прозрачные результаты;
- возможность легко доращивать пайплайн и UI без «склейки» логики.

