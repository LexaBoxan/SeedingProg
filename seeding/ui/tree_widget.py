"""Дерево слоёв для отображения оригиналов и найденных объектов."""

from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt


class LayerTreeWidget(QTreeWidget):
    """Простой QTreeWidget для отображения иерархии изображений."""

    def __init__(self) -> None:
        """Конструктор дерева слоёв."""
        super().__init__()
        self.setHeaderLabels(["Название", "Описание"])
        # Отключаем редактирование элементов, чтобы клики по ним
        # вызывали открытие изображений, а не режим редактирования
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def add_root_item(self, name, description, index, image_type, image):
        """
        Добавляет корневой элемент (страницу, оригинал и т.д.) в дерево.
        """
        root = QTreeWidgetItem(self)
        root.setText(0, name)
        root.setText(1, description)
        # Сохраняем изображение и индекс внутри UserRole
        root.setData(0, Qt.UserRole, {"index": index, "type": image_type})
        # Убираем возможность редактирования названия элемента
        root.setFlags(root.flags() & ~Qt.ItemIsEditable)
        self.addTopLevelItem(root)
        return root

    def add_child_item(
        self, parent, name, description, parent_index, index, image_type, image
    ):
        """
        Добавляет дочерний элемент к выбранному родителю.
        """
        child = QTreeWidgetItem(parent)
        child.setText(0, name)
        child.setText(1, description)
        # Тут parent_index — это индекс оригинального изображения, index — это индекс сеянца (crop-а)
        child.setData(
            0,
            Qt.UserRole,
            {"type": "seeding", "parent_index": parent_index, "index": index},
        )
        # Убираем возможность редактирования названия элемента
        child.setFlags(child.flags() & ~Qt.ItemIsEditable)
        parent.addChild(child)
        return child

    def add_class_item(
        self,
        parent: QTreeWidgetItem,
        name: str,
        description: str,
        parent_index: int,
        seeding_index: int,
        class_index: int,
    ) -> QTreeWidgetItem:
        """Добавляет подпункт классификации под выбранным сеянцем.

        В элементе сохраняются индексы родительского изображения,
        сеянца и класса, что позволяет при клике отображать
        соответствующий вырез изображения.
        """

        child = QTreeWidgetItem(parent)
        child.setText(0, name)
        child.setText(1, description)
        child.setData(
            0,
            Qt.UserRole,
            {
                "type": "class",
                "parent_index": parent_index,
                "seeding_index": seeding_index,
                "class_index": class_index,
            },
        )
        # Убираем возможность редактирования названия элемента
        child.setFlags(child.flags() & ~Qt.ItemIsEditable)
        parent.addChild(child)
        return child
