from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ..utils import HOVER_COLOR, getStandardFont

class DelegateSize(QStyledItemDelegate):
    def __init__(self,parent):
        super(DelegateSize, self).__init__(parent)

    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)
        r = option.rect.height()/4

        if self.parent().model().getCheckStateByTableIndex(index):

            painter.save()
            pen = painter.pen()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            pen.setColor(QColor("darkgrey"))
            pen.setWidthF(0.5)
            painter.setPen(pen)           
            centerPoint = option.rect.center()
            painter.drawEllipse(QPointF(centerPoint),r,r)
            rect = option.rect
            rect.adjust(8,8,-8,-8)
            painter.drawText(rect,Qt.AlignmentFlag.AlignCenter, str(self.parent().model().getSize(index)))
            painter.restore() 
            
        elif self.parent().mouseOverItem is not None and index.row() == self.parent().mouseOverItem:
            painter.save()
            pen = painter.pen()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            pen.setColor(QColor("darkgrey"))
            pen.setWidthF(0.5)
            painter.setPen(pen)
            centerPoint = option.rect.center()
            painter.drawEllipse(QPointF(centerPoint),r,r)
            painter.restore()        

class DelegateColor(QStyledItemDelegate):
    def __init__(self,parent):
        super(DelegateColor, self).__init__(parent)

    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)
       # print(self.parent().mouseOverItem)
        centerPoint = option.rect.center()
        r = option.rect.height()/4
        pen = painter.pen()
        pen.setColor(QColor("darkgrey"))
        pen.setWidthF(0.5)
        painter.setPen(pen)

        painter.save()
        color = self.parent().model().getColor(index)
        if isinstance(color,str):
            color = QColor(color)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        brush = QBrush(color)
        #painter.setPen(Qt.NoPen)
        painter.setBrush(brush)
        
        painter.drawEllipse(QPointF(centerPoint),r,r)
        painter.restore()
    