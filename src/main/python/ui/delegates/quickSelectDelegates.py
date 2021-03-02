from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ..utils import HOVER_COLOR

class DelegateSize(QStyledItemDelegate):
    def __init__(self,parent):
        super(DelegateSize, self).__init__(parent)

    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)
        r = option.rect.height()/4

        if self.parent().model().getCheckStateByTableIndex(index):

            painter.save()
            pen = painter.pen()
            painter.setRenderHint(QPainter.Antialiasing,True)
            pen.setColor(QColor("darkgrey"))
            pen.setWidthF(0.5)
            painter.setPen(pen)           
            centerPoint = option.rect.center()
            painter.drawEllipse(centerPoint,r,r)
            rect = option.rect
            rect.adjust(8,8,-8,-8)
            painter.drawText(rect,Qt.AlignCenter, str(self.parent().model().getSize(index)))
            painter.restore() 
            
        elif self.parent().mouseOverItem is not None and index.row() == self.parent().mouseOverItem:
            painter.save()
            pen = painter.pen()
            painter.setRenderHint(QPainter.Antialiasing,True)
            pen.setColor(QColor("darkgrey"))
            pen.setWidthF(0.5)
            painter.setPen(pen)
            centerPoint = option.rect.center()
            painter.drawEllipse(centerPoint,r,r)
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

        color = self.parent().model().getColor(index)
        if isinstance(color,str):
            color = QColor(color)
        painter.setRenderHint(QPainter.Antialiasing,True)
        brush = QBrush(color)
        #painter.setPen(Qt.NoPen)
        painter.setBrush(brush)
        
        painter.drawEllipse(centerPoint,r,r)
        


        # elif self.parent().mouseOverItem is not None and index.row() == self.parent().mouseOverItem:
        #     painter.save()
        #     pen = painter.pen()
        #     painter.setRenderHint(QPainter.Antialiasing,True)
        #     brush = QBrush(QColor("lightgrey"))
        #     pen.setColor(Qt.transparent)
        #     painter.setPen(pen)
        #     painter.setBrush(brush)
            
        #     painter.drawEllipse(centerPoint,r,r)
        #     painter.restore()



class ItemDelegate(QStyledItemDelegate):
    def __init__(self,parent):
        super(ItemDelegate,self).__init__(parent)
    

    def paint(self, painter, option, index):
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        rect = option.rect
        if self.parent().mouseOverItem is not None and index.row() == self.parent().mouseOverItem:
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.NoPen)
            painter.drawRect(option.rect)
            self.addText(index,painter,rect)

        # elif self.parent().mouseOverItem is not None and index.row() == self.parent().mouseOverItem:
        #     b = QBrush(QColor("lightgrey"))
        #     painter.setBrush(b)
        #     painter.setPen(Qt.NoPen)
        #     painter.drawRect(option.rect)
        #     self.addText(index,painter,rect)
        
        else:
            self.addText(index,painter,rect)
    
    def addText(self,index,painter,rect):
        ""
        painter.setPen(QPen(QColor("black")))
        rect.adjust(9,0,0,0)
        painter.drawText(rect,   Qt.AlignVCenter | Qt.AlignLeft, self.parent().model().data(index,Qt.DisplayRole))
       
    def setEditorData(self,editor,index):
        
        editor.setAutoFillBackground(True)
        editor.setText(self.parent().model().data(index,Qt.DisplayRole))