"""Resizable rectangle item for interactive bounding boxes."""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem, QStyleOptionGraphicsItem, QWidget


class BBoxItem(QGraphicsRectItem):
    """QGraphicsRectItem with resize handles linked to ObjectImage."""

    HANDLE_SIZE = 8.0

    def __init__(
        self,
        rect: QRectF,
        obj,
        parent: QGraphicsItem | None = None,
        color=Qt.green,
        offset=(0, 0),
    ):
        super().__init__(rect, parent)
        self.obj = obj
        self.offset = offset
        self.setPen(QPen(color, 2))
        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self._editable = False
        self._handle = None
        self._handles = {}
        self._update_handles()

    # ------------------------------------------------------------------
    # handle utilities
    def _update_handles(self) -> None:
        """Recalculate handle rectangles."""
        r = self.rect()
        s = self.HANDLE_SIZE
        self._handles = {
            "tl": QRectF(r.x() - s / 2, r.y() - s / 2, s, s),
            "tr": QRectF(r.right() - s / 2, r.y() - s / 2, s, s),
            "bl": QRectF(r.x() - s / 2, r.bottom() - s / 2, s, s),
            "br": QRectF(r.right() - s / 2, r.bottom() - s / 2, s, s),
        }

    def setEditable(self, state: bool) -> None:
        """Enable or disable editing."""
        self._editable = state
        self.setFlag(QGraphicsItem.ItemIsMovable, state)
        self.update()

    # ------------------------------------------------------------------
    # painting
    def paint(self, painter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None):
        super().paint(painter, option, widget)
        if self._editable:
            painter.setBrush(Qt.white)
            for handle_rect in self._handles.values():
                painter.drawRect(handle_rect)

    # ------------------------------------------------------------------
    # mouse events
    def mousePressEvent(self, event):
        if self._editable:
            for name, rect in self._handles.items():
                if rect.contains(event.pos()):
                    self._handle = name
                    break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._editable and self._handle:
            r = self.rect()
            pos = event.pos()
            if self._handle == "tl":
                r.setTopLeft(pos)
            elif self._handle == "tr":
                r.setTopRight(pos)
            elif self._handle == "bl":
                r.setBottomLeft(pos)
            elif self._handle == "br":
                r.setBottomRight(pos)
            self.setRect(r)
        else:
            super().mouseMoveEvent(event)
        if self._editable:
            self._update_handles()
            self.update_bbox()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._handle = None
        if self._editable:
            self._update_handles()
            self.update_bbox()

    # ------------------------------------------------------------------
    def update_bbox(self) -> None:
        """Update bbox in linked ObjectImage."""
        r = self.rect().normalized()
        ox, oy = self.offset
        self.obj.bbox = (
            int(r.left()) + ox,
            int(r.top()) + oy,
            int(r.right()) + ox,
            int(r.bottom()) + oy,
        )
