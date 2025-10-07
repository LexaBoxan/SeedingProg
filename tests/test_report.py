import os
import numpy as np
from seeding.models.data_models import OriginalImage, ObjectImage, AllClassImage
from seeding.report import create_pdf_report, _annotate_image

def test_create_pdf_report(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    obj = ObjectImage(class_name="Seeding", confidence=0.9, image=[img], bbox=(1, 2, 3, 4))
    data = OriginalImage(file_path=str(tmp_path / "image.png"), images=[img], class_object_image=[[obj]])
    output = tmp_path / "report.pdf"
    create_pdf_report(data, str(output))
    assert output.is_file() and output.stat().st_size > 0


def test_annotate_image_with_class_bbox():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cls = AllClassImage("part", 0.8, img, bbox=(1, 1, 3, 3))
    obj = ObjectImage(
        class_name="Seeding",
        confidence=0.9,
        image=[img],
        bbox=(1, 1, 5, 5),
        image_all_class=[cls],
    )
    annotated = _annotate_image(img, [obj])
    assert (annotated[2, 2] == np.array([255, 0, 0])).all()
