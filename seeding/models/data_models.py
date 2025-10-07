"""Dataclasses для хранения изображений и результатов детекции."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
from PIL import Image


@dataclass
class AllClassImage:
    """
    Класс для хранения информации о выделенном классе на изображении.

    Attributes:
        class_name (str): Название класса (например, тип растения).
        confidence (float): Уверенность модели в принадлежности к классу (0.0 - 1.0).
        image (Union[np.ndarray, Image.Image]): Картинка участка, соответствующего данному классу.
    """

    class_name: str
    confidence: float
    image: Union[np.ndarray, Image.Image]
    bbox: tuple | None = None  # (x1, y1, x2, y2) относит. к кропу сеянца


@dataclass
class ObjectImage:
    """
    Класс для хранения информации о выделенном объекте на изображении.

    Attributes:
        class_name (str): Название класса объекта.
        confidence (float): Уверенность в детекции объекта.
        image (List[Union[np.ndarray, Image.Image]]): Список изображений, соответствующих объекту.
        image_all_class (Optional[List[AllClassImage]]): Подклассы, выделенные внутри объекта.
    """

    class_name: str
    confidence: float
    image: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    image_all_class: Optional[List[AllClassImage]] = None
    bbox: tuple = None  # (x1, y1, x2, y2)
    rotation_k: int = 0  # Поворот, применённый к crop (значение k для np.rot90)


@dataclass
class OriginalImage:
    """
    Класс для хранения всей информации о загруженном исходном изображении и связанных с ним данных.

    Attributes:
        file_path (str): Путь к исходному файлу.
        images (List[Union[np.ndarray, Image.Image]]): Список исходных изображений (например, страницы PDF).
        masks (List[Union[np.ndarray, Image.Image]]): Список масок для изображений.
        final_images (List[Union[np.ndarray, Image.Image]]): Список итоговых изображений после обработки.
        class_object_image (Optional[List[ObjectImage]]): Список объектов, выделенных на изображениях.
    """

    file_path: str = ""
    images: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    masks: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    final_images: List[Union[np.ndarray, Image.Image]] = field(default_factory=list)
    class_object_image: Optional[List[ObjectImage]] = None
