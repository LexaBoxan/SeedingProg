"""Основное окно приложения."""

import logging
import os

import cv2
import fitz
import numpy as np
from PyQt5.QtCore import QPoint, Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QIcon, QImage, QPixmap, QTransform

from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QGroupBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QStyle,
    QToolBar,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from seeding.config import ROTATE_K, DEFAULT_CLASSIFY_WEIGHTS_PATH
from seeding.models.data_models import AllClassImage, ObjectImage, OriginalImage
from seeding.utils import simple_nms, rotate_bbox
from .tree_widget import LayerTreeWidget
from .bbox_item import BBoxItem

logger = logging.getLogger(__name__)

# Ожидаемые классы для модели классификации
# Используем список без привязки к индексам, чтобы лишь проверять состав классов
EXPECTED_CLASSIFY_NAMES = ["flower", "root", "stem"]


class DraggableScrollArea(QScrollArea):
    """
    ScrollArea c возможностью перетаскивания средней кнопкой мыши.
    """

    def __init__(self, parent=None):
        """Конструктор виджета с поддержкой перетаскивания."""
        super().__init__(parent)
        self._drag_active = False
        self._drag_start_pos = QPoint()
        self._scroll_start_pos = QPoint()

    def mousePressEvent(self, event):
        """Начинает перетаскивание при нажатии средней кнопкой мыши."""
        if event.button() == Qt.MiddleButton:
            self._drag_active = True
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_start_pos = event.pos()
            self._scroll_start_pos = QPoint(
                self.horizontalScrollBar().value(), self.verticalScrollBar().value()
            )
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Перемещает содержимое при активном перетаскивании."""
        if self._drag_active:
            delta = event.pos() - self._drag_start_pos
            self.horizontalScrollBar().setValue(self._scroll_start_pos.x() - delta.x())
            self.verticalScrollBar().setValue(self._scroll_start_pos.y() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Завершает перетаскивание."""
        if event.button() == Qt.MiddleButton:
            self._drag_active = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)


class DetectionWorker(QThread):
    """Worker для выполнения детекции в отдельном потоке."""

    result_ready = pyqtSignal(int, object)

    def __init__(self, index: int, image: np.ndarray, model: YOLO | None = None, weights_path: str | None = None):
        super().__init__()
        self.index = index
        self.image = image
        self.model = model
        self.weights_path = weights_path

    def run(self) -> None:  # pragma: no cover - поток
        model = self.model
        if model is None and self.weights_path:
            model = YOLO(self.weights_path)
        if model is None:
            return
        results = model(self.image)
        self.result_ready.emit(self.index, results)


class ImageEditor(QMainWindow):
    """
    Главное окно приложения для работы с изображениями и PDF.

    Позволяет загружать файлы, управлять слоями и искать сеянцы при помощи YOLOv8.
    """

    def __init__(self, weights_path: str):
        """Инициализирует окно и загружает модель.

        Args:
            weights_path: Путь к файлу весов YOLOv8.
        """
        super().__init__()
        self.setWindowTitle("Современный UI для работы с изображениями")
        self.setGeometry(100, 100, 1200, 800)
        self.zoom_factor = 1.0
        self.image_storage = OriginalImage()
        self.weights_path = weights_path
        self.model = YOLO(weights_path)
        self.classify_model = None

        self.init_ui()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def init_ui(self):
        """Создаёт все основные виджеты интерфейса."""
        self.create_menu()
        self.create_toolbars()
        self.create_central_widget()
        self.create_right_panel()

    def create_menu(self):
        """Создаёт меню приложения с пунктом открытия файла."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        open_action = QAction("Открыть файл", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

    def _update_action_states(self) -> None:
        """Включает или отключает действия, зависящие от наличия изображения."""
        has_image = bool(self.image_storage.images)
        for action in (
            self.rotate_action,
            self.seedlings_action,
            self.find_all_seedlings_action,
            self.classify_action,
            self.report_action,
            self.save_action,
        ):
            action.setEnabled(has_image)

    def create_toolbars(self):
        """Создаёт боковую панель инструментов."""
        toolbar = QToolBar("Toolbar", self)
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setMovable(False)
        toolbar.setFixedWidth(150)

        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)

        style = self.style()

        self.mask_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogNewFolder), "Создать маску", self
        )
        self.mask_action.triggered.connect(self.create_mask)
        toolbar.addAction(self.mask_action)

        self.seedlings_action = QAction(
            style.standardIcon(QStyle.SP_MediaPlay), "Найти сеянцы", self
        )
        self.seedlings_action.triggered.connect(self.find_seedlings)
        toolbar.addAction(self.seedlings_action)

        self.find_all_seedlings_action = QAction(
            style.standardIcon(QStyle.SP_DialogYesButton), "Найти все сеянцы", self
        )
        self.find_all_seedlings_action.triggered.connect(self.find_all_seedlings)
        toolbar.addAction(self.find_all_seedlings_action)

        self.classify_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogDetailedView), "Классификация", self
        )
        self.classify_action.triggered.connect(self.classify)
        toolbar.addAction(self.classify_action)

        self.rotate_action = QAction(
            style.standardIcon(QStyle.SP_BrowserReload), "Повернуть на 90°", self
        )
        self.rotate_action.triggered.connect(self.rotate_image)
        toolbar.addAction(self.rotate_action)

        toolbar.addSeparator()

        self.report_action = QAction(
            style.standardIcon(QStyle.SP_FileDialogContentsView), "Создать отчет", self
        )
        self.report_action.triggered.connect(self.create_report)
        toolbar.addAction(self.report_action)

        toolbar.addSeparator()

        self.zoom_in_action = QAction(
            style.standardIcon(QStyle.SP_ArrowUp), "Приблизить", self
        )
        self.zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction(
            style.standardIcon(QStyle.SP_ArrowDown), "Отдалить", self
        )
        self.zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(self.zoom_out_action)

        self.fit_action = QAction(
            style.standardIcon(QStyle.SP_DesktopIcon), "Вписать", self
        )
        self.fit_action.triggered.connect(self.fit_to_window)
        toolbar.addAction(self.fit_action)

        toolbar.addSeparator()

        self.save_action = QAction(
            style.standardIcon(QStyle.SP_DialogSaveButton),
            "Сохранить изменения",
            self,
        )
        self.save_action.triggered.connect(self.save_changes)
        toolbar.addAction(self.save_action)

        self._update_action_states()

    def create_central_widget(self):
        """Создаёт центральную область отображения изображений."""
        self.scroll_area = DraggableScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.image_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.image_item)
        self.rect_items = {}

        self.scroll_area.setWidget(self.graphics_view)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.scroll_area)
        self.setCentralWidget(self.splitter)

    def create_right_panel(self):
        """Создаёт правую панель с деревом слоёв."""
        self.right_panel = QGroupBox("Слои")
        self.right_panel.setMinimumWidth(200)
        layout = QVBoxLayout()
        self.tree_widget = LayerTreeWidget()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.tree_widget)
        layout.addWidget(scroll_area)

        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)

        self.right_panel.setLayout(layout)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setCollapsible(1, False)

    def on_tree_item_clicked(self, item, column):
        """Обрабатывает выбор элемента в дереве слоёв."""
        item_data = item.data(0, Qt.UserRole)
        if item_data:
            if item_data["type"] in ("original", "pdf"):
                idx = item_data["index"]
                self._active_image_index = idx
                self.display_image_with_boxes(idx)
            elif item_data["type"] == "seeding":
                parent_idx = item_data["parent_index"]
                seed_idx = item_data["index"]
                self._active_image_index = parent_idx
                self.display_seeding_with_boxes(parent_idx, seed_idx)
            elif item_data["type"] == "class":
                parent_idx = item_data["parent_index"]
                seed_idx = item_data["seeding_index"]
                class_idx = item_data["class_index"]
                self._active_image_index = parent_idx
                self.display_class_image(parent_idx, seed_idx, class_idx)
            else:
                return

    def open_image(self) -> None:
        """Открывает диалог выбора файла и загружает изображение или PDF."""
        self.image_storage = OriginalImage()
        self._update_action_states()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть изображение или PDF",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;PDF Files (*.pdf);;All Files (*)",
        )
        if file_name:
            self.image_storage.file_path = file_name
            self.image_storage.images.clear()
            self.tree_widget.clear()

            if file_name.lower().endswith(".pdf"):
                self.load_pdf(file_name)
            else:
                image = self.load_image(file_name)
                if image is not None:
                    self.image_storage.images.append(image)
                    self.display_image(image)
                    self._active_image_index = 0
                    self.tree_widget.add_root_item(
                        "Оригинал", "Исходное изображение", 0, "original", image
                    )

            # Обязательно инициализируем пустые списки для найденных объектов
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

        self._update_action_states()

    def load_image(self, file_name: str) -> np.ndarray | None:
        """Загружает изображение с диска."""
        try:
            image = cv2.imread(file_name)
            return image
        except Exception as e:
            logger.error("Ошибка при загрузке изображения: %s", e)
            return None

    def load_pdf(self, pdf_path: str) -> None:
        """Загружает все страницы PDF как изображения."""
        try:
            doc = fitz.open(pdf_path)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, doc.page_count)
            self.progress_bar.setValue(0)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(4, 4)  # 2x масштаб
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img = img[:, :, :3].copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.image_storage.images.append(img)
                # Для первой страницы — показать в QLabel
                if page_num == 0:
                    self.display_image(img)
                # Добавить в дерево
                self.tree_widget.add_root_item(
                    f"Стр. {page_num + 1}", "Страница PDF", page_num, "pdf", img
                )
                self.progress_bar.setValue(page_num + 1)
            doc.close()

            # Инициализация class_object_image для всех страниц
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

        except Exception as e:
            logger.error("Ошибка при загрузке PDF: %s", e)

    def display_image(self, image: np.ndarray) -> None:
        """Отображает переданное изображение в центральной области."""
        if image is None or not isinstance(image, np.ndarray):
            return
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return

        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qformat = QImage.Format_RGB888
        elif len(image.shape) == 2:
            image_rgb = image
            bytes_per_line = width
            qformat = QImage.Format_Grayscale8
        else:
            return

        image_rgb = np.ascontiguousarray(image_rgb)
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, qformat)
        self._original_image = image
        self._original_pixmap = QPixmap.fromImage(q_image)

        self.graphics_scene.clear()
        self.image_item = self.graphics_scene.addPixmap(self._original_pixmap)
        self.rect_items = {}

        scroll_size = self.scroll_area.viewport().size()
        ratio_w = scroll_size.width() / self._original_pixmap.width()
        ratio_h = scroll_size.height() / self._original_pixmap.height()
        self.min_fit_zoom = min(ratio_w, ratio_h, 1.0)
        self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def zoom_in(self) -> None:
        """Увеличивает изображение."""
        self.zoom_factor *= 1.25
        self.update_image_zoom()

    def zoom_out(self) -> None:
        """Уменьшает изображение."""
        self.zoom_factor /= 1.25
        if self.zoom_factor < self.min_fit_zoom:
            self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def fit_to_window(self) -> None:
        """Масштабирует изображение так, чтобы оно поместилось в окно."""
        self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()

    def update_image_zoom(self) -> None:
        """Применяет текущий масштаб к изображению."""
        if hasattr(self, "_original_pixmap"):
            transform = QTransform()
            transform.scale(self.zoom_factor, self.zoom_factor)
            self.graphics_view.setTransform(transform)
            self.graphics_scene.setSceneRect(
                0,
                0,
                self._original_pixmap.width(),
                self._original_pixmap.height(),
            )

    def rotate_image(self) -> None:
        """Поворачивает выбранное изображение или crop на 90 градусов."""
        self._update_action_states()
        selected_item = self.tree_widget.currentItem()
        if selected_item is None:
            logger.warning("rotate_image: Нет выбранного элемента в дереве")
            return

        item_data = selected_item.data(0, Qt.UserRole)
        if not item_data:
            logger.warning("rotate_image: Нет данных для выбранного элемента")
            return

        if item_data["type"] in ("original", "pdf"):
            idx = item_data["index"]
            image = self.image_storage.images[idx]
            if image is None:
                logger.warning("rotate_image: Оригинал отсутствует")
                return
            rotated = np.rot90(image, k=ROTATE_K)
            self.image_storage.images[idx] = rotated
            logger.info("rotate_image: Изображение %s повернуто", idx)
            self.display_image(rotated)

        elif item_data["type"] == "seeding":
            parent_idx = item_data["parent_index"]
            seed_idx = item_data["index"]
            obj = self.image_storage.class_object_image[parent_idx][seed_idx]
            if not obj.image or obj.image[0] is None:
                logger.warning("rotate_image: Crop пустой")
                return
            crop = obj.image[0]
            rotated = np.rot90(crop, k=ROTATE_K)
            self.image_storage.class_object_image[parent_idx][seed_idx].image[
                0
            ] = rotated
            obj.rotation_k = (obj.rotation_k + ROTATE_K) % 4
            logger.info(
                "rotate_image: Crop %s (от оригинала %s) повернут",
                seed_idx,
                parent_idx,
            )
            self.display_image(rotated)
        else:
            logger.warning("rotate_image: Неизвестный тип данных")
            return

    def create_mask(self) -> None:
        """Создание маски (функциональность пока не реализована)."""
        logger.info("Создание маски — пока не реализовано")

    def _on_detection_start(self) -> None:
        """Показывает прогресс-бар при запуске worker."""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

    def _on_detection_finished(self) -> None:
        """Скрывает прогресс-бар после завершения worker."""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _on_detection_result(self, index: int, results) -> None:
        """Обрабатывает результаты детекции и обновляет дерево."""
        image = self.image_storage.images[index]
        try:
            boxes: list[list[int]] = []
            scores: list[float] = []
            class_boxes_data: list[dict] = []
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = results[0].names[class_id]
                if class_name == "Seeding":
                    score = float(box.conf)
                    x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    h, w = image.shape[:2]
                    x1, x2 = max(0, x1), min(x2, w)
                    y1, y2 = max(0, y1), min(y2, h)
                    if x2 <= x1 or y2 <= y1:
                        logger.debug(
                            "find_seedlings: пропускаем некорректный bbox %s",
                            (x1, y1, x2, y2),
                        )
                        continue

                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_boxes_data.append(
                        {
                            "class_name": class_name,
                            "score": score,
                            "coords": (x1, y1, x2, y2),
                        }
                    )

            logger.info(
                "find_seedlings: найдено %s боксов, запускаем NMS", len(boxes)
            )
            indices = simple_nms(boxes, scores, iou_threshold=0.4)
            logger.info(
                "find_seedlings: после NMS осталось %s боксов", len(indices)
            )

            self.image_storage.class_object_image[index] = []
            parent_item = self.tree_widget.topLevelItem(index)
            if parent_item is not None:
                for i in reversed(range(parent_item.childCount())):
                    parent_item.takeChild(i)
            for i_out, i in enumerate(indices):
                data = class_boxes_data[i]
                x1, y1, x2, y2 = data["coords"]
                crop = image[y1:y2, x1:x2].copy()
                rotation_k = 0
                if crop.shape[1] > crop.shape[0]:
                    crop = np.rot90(crop, k=ROTATE_K)
                    rotation_k = ROTATE_K
                obj = ObjectImage(
                    class_name=data["class_name"],
                    confidence=data["score"],
                    image=[crop],
                    bbox=(x1, y1, x2, y2),
                    rotation_k=rotation_k,
                )
                self.image_storage.class_object_image[index].append(obj)
                self.tree_widget.add_child_item(
                    parent_item,
                    f"Seeding{i_out + 1}",
                    f"Уверенность: {data['score']:.2f}",
                    index,
                    i_out,
                    "seeding",
                    crop,
                )
            self.display_image_with_boxes(index)
            logger.info("find_seedlings: завершено")
        except Exception as e:  # pragma: no cover - логирование
            logger.error("Ошибка во время NMS или обработки результатов: %s", e)

    def find_seedlings(self) -> None:
        """Запускает модель YOLOv8 для поиска сеянцев на текущем изображении.

        Результаты проходят через простую процедуру NMS. Каждая найденная
        область добавляется в хранилище `image_storage` и отображается в дереве
        слоёв. Если ширина вырезанного участка больше его высоты, изображение
        поворачивается на 90 градусов для вертикальной ориентации.
        """
        self._update_action_states()
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

        logger.info("find_seedlings: start")
        current_index = getattr(self, "_active_image_index", 0)
        logger.debug("find_seedlings: current_index = %s", current_index)

        if not self.image_storage.images:
            logger.warning("find_seedlings: Нет изображений для обработки")
            return

        image = self.image_storage.images[current_index]
        if image is None:
            logger.warning("find_seedlings: Текущее изображение пустое")
            return

        self.worker = DetectionWorker(
            index=current_index,
            image=image,
            model=self.model,
            weights_path=self.weights_path,
        )
        self.worker.started.connect(self._on_detection_start)
        self.worker.result_ready.connect(self._on_detection_result)
        self.worker.finished.connect(self._on_detection_finished)
        self.worker.start()

    def find_all_seedlings(self) -> None:
        """Запускает поиск сеянцев на всех изображениях без падений.

        Ранее метод вызывал :meth:`find_seedlings`, который стартовал
        асинхронный `QThread` для каждой страницы. При последовательном
        обходе изображений это приводило к одновременному запуску множества
        потоков и приложению было сложно корректно обновлять прогресс‑бар,
        что могло завершаться крашем. Теперь детекция выполняется
        синхронно в основном потоке: результаты каждой страницы
        обрабатываются сразу после получения, а индикатор прогресса
        обновляется последовательно.
        """

        self._update_action_states()
        if not self.image_storage.images:
            logger.warning("find_all_seedlings: Нет изображений")
            return

        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = [
                [] for _ in range(len(self.image_storage.images))
            ]

        total = len(self.image_storage.images)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)

        for idx, image in enumerate(self.image_storage.images):
            self._active_image_index = idx
            self.progress_bar.setValue(idx)

            results = self.model(image)
            self._on_detection_result(idx, results)

            self.progress_bar.setValue(idx + 1)

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        logger.info("find_all_seedlings: завершено")

    def display_image_with_boxes(self, idx: int) -> None:
        """Отображает изображение с нанесёнными рамками объектов."""
        image = self.image_storage.images[idx].copy()
        self.display_image(image)
        if (
            self.image_storage.class_object_image
            and len(self.image_storage.class_object_image) > idx
        ):
            for obj_idx, obj in enumerate(self.image_storage.class_object_image[idx]):
                if obj.bbox:
                    x1, y1, x2, y2 = obj.bbox
                    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    rect_item = BBoxItem(rect, obj)
                    rect_item.setEditable(True)
                    self.graphics_scene.addItem(rect_item)
                    self.rect_items[(idx, obj_idx)] = rect_item



    def display_seeding_with_boxes(self, parent_idx: int, seed_idx: int) -> None:
        """Отображает crop сеянца с его классификационными боксами."""
        if (
            not self.image_storage.class_object_image
            or parent_idx >= len(self.image_storage.class_object_image)
            or seed_idx >= len(self.image_storage.class_object_image[parent_idx])
        ):
            return

        obj = self.image_storage.class_object_image[parent_idx][seed_idx]
        if not obj.image:
            return

        crop_img = obj.image[0].copy()
        self.display_image(crop_img)

        if obj.image_all_class:
            for cls_idx, cls_obj in enumerate(obj.image_all_class):
                if cls_obj.bbox:
                    lx1, ly1, lx2, ly2 = cls_obj.bbox
                    rect = QRectF(lx1, ly1, lx2 - lx1, ly2 - ly1)
                    rect_item = BBoxItem(rect, cls_obj, color=Qt.red)
                    rect_item.setEditable(True)
                    self.graphics_scene.addItem(rect_item)
                    self.rect_items[(parent_idx, seed_idx, cls_idx)] = rect_item

    def display_class_image(
        self, parent_idx: int, seed_idx: int, class_idx: int
    ) -> None:
        """Отображает вырез конкретного класса внутри сеянца."""
        if (
            not self.image_storage.class_object_image
            or parent_idx >= len(self.image_storage.class_object_image)
            or seed_idx >= len(self.image_storage.class_object_image[parent_idx])
        ):
            return

        obj = self.image_storage.class_object_image[parent_idx][seed_idx]
        if not obj.image_all_class or class_idx >= len(obj.image_all_class):
            return

        cls_obj = obj.image_all_class[class_idx]
        if cls_obj.image is None:
            return

        self.display_image(cls_obj.image)

    def save_changes(self) -> None:
        """Пересохраняет crop-изображения после изменения рамок."""
        if not self.image_storage.images or not self.image_storage.class_object_image:
            return
        for img_idx, objects in enumerate(self.image_storage.class_object_image):
            if img_idx >= len(self.image_storage.images):
                continue
            base_img = self.image_storage.images[img_idx]
            for obj in objects:
                if obj.bbox:
                    x1, y1, x2, y2 = obj.bbox
                    crop = base_img[y1:y2, x1:x2].copy()
                    if getattr(obj, "rotation_k", 0):
                        crop = np.rot90(crop, k=obj.rotation_k)
                    obj.image = [crop]
                if obj.image_all_class:
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
                            if obj.bbox:
                                gx1 = obj.bbox[0] + ux1
                                gy1 = obj.bbox[1] + uy1
                                gx2 = obj.bbox[0] + ux2
                                gy2 = obj.bbox[1] + uy2
                            else:
                                gx1, gy1, gx2, gy2 = ux1, uy1, ux2, uy2
                            part = base_img[gy1:gy2, gx1:gx2].copy()
                            if k:
                                part = np.rot90(part, k=k)
                            cls.image = part
        logger.info("save_changes: обновлённые координаты сохранены")

    def classify(self) -> None:
        """Определяет для каждого сеянца классы: flower, root и stem."""
        self._update_action_states()

        if not self.image_storage.class_object_image:
            logger.warning("classify: Нет объектов для классификации")
            return

        if self.classify_model is None:
            try:
                self.classify_model = YOLO(str(DEFAULT_CLASSIFY_WEIGHTS_PATH))
                model_names = self.classify_model.names
                loaded_names = (
                    list(model_names.values())
                    if isinstance(model_names, dict)
                    else list(model_names)
                )
                if set(loaded_names) != set(EXPECTED_CLASSIFY_NAMES):  # pragma: no cover - логирование
                    logger.error(
                        "classify: unexpected class names %s, expected %s",
                        loaded_names,
                        EXPECTED_CLASSIFY_NAMES,
                    )
                    self.classify_model = None
                    return
            except Exception as e:  # pragma: no cover - логирование
                logger.error("Не удалось загрузить модель классификации: %s", e)
                self.classify_model = None
                return

        for img_idx, objects in enumerate(self.image_storage.class_object_image):
            page_item = self.tree_widget.topLevelItem(img_idx)
            for obj_idx, obj in enumerate(objects):
                if not obj.image:
                    continue
                crop = obj.image[0]
                try:
                    result = self.classify_model(crop)[0]
                except Exception as e:  # pragma: no cover - логирование
                    logger.error("Ошибка классификации: %s", e)
                    continue

                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    logger.debug("classify: не найден класс для объекта %s", obj_idx)
                    continue

                detections = list(
                    zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())
                )
                detections.sort(key=lambda x: x[1], reverse=True)

                obj.image_all_class = []
                seeding_item = page_item.child(obj_idx) if page_item is not None else None
                if seeding_item is not None:
                    for i in reversed(range(seeding_item.childCount())):
                        seeding_item.takeChild(i)

                names = self.classify_model.names
                for cls_idx, (cls_id, conf, coords) in enumerate(detections):
                    cls_id = int(cls_id)
                    conf = float(conf)
                    x1, y1, x2, y2 = map(int, coords)
                    part_img = crop[y1:y2, x1:x2].copy()
                    class_name = (
                        names.get(cls_id, str(cls_id))
                        if isinstance(names, dict)
                        else (
                            names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
                        )
                    )

                    local_bbox = (x1, y1, x2, y2)

                    obj.image_all_class.append(
                        AllClassImage(
                            class_name=class_name,
                            confidence=conf,
                            image=part_img,
                            bbox=local_bbox,
                        )
                    )

                    if seeding_item is not None:
                        self.tree_widget.add_class_item(
                            seeding_item,
                            class_name,
                            f"Уверенность: {conf:.2f}",
                            img_idx,
                            obj_idx,
                            cls_idx,
                        )

        active_idx = getattr(self, "_active_image_index", 0)
        self.display_image_with_boxes(active_idx)
        logger.info("classify: завершено")

    def create_report(self) -> None:
        """Создаёт PDF-отчёт по текущим результатам детекции."""
        self._update_action_states()
        if not self.image_storage.images:
            logger.warning("create_report: Нет данных для отчёта")
            return

        base_path, _ = os.path.splitext(self.image_storage.file_path)
        output_path = base_path + "_report.pdf"
        try:
            from ..report import create_pdf_report

            create_pdf_report(self.image_storage, output_path)
            logger.info("Отчёт сохранён: %s", output_path)
        except Exception as e:
            logger.error("Ошибка при создании отчёта: %s", e)
