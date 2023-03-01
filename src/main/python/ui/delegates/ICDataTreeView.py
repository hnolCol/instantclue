from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ..utils import HOVER_COLOR, getStandardFont

class DeleteDelegate(QStyledItemDelegate):
    def __init__(self,parent, highLightColumn = 1):
        super(DeleteDelegate,self).__init__(parent)
        self.highLightColumn = highLightColumn

    def sizeHint(self):
        ""
        return QSize(10,10)

    def paint(self, painter, option, index):
        if self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            centerPoint = option.rect.center()
            background = QRect(option.rect)
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(background)

            pen = QPen(QColor("darkgrey" if self.parent().focusColumn != self.highLightColumn else "#B84D29"))
            pen.setWidthF(2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            x0 = centerPoint.x()
            y0 = centerPoint.y()
            h = option.rect.height()/8
            painter.drawLine(QLineF(x0-h,y0-h,x0+h,y0+h))
            painter.drawLine(QLineF(x0+h,y0-h,x0-h,y0+h))
        
        elif self.parent().selectionModel().isSelected(self.parent().model().index(index.row(),0)):
           b = QBrush(QColor("lightgrey"))
           painter.setBrush(b)
           painter.setPen(Qt.PenStyle.NoPen)
           painter.drawRect(option.rect)


class GroupDelegate(QStyledItemDelegate):
    def __init__(self,parent, highLightColumn = 1):
        super(GroupDelegate,self).__init__(parent)
        self.highLightColumn = highLightColumn

    def sizeHint(self):
        ""
        return QSize(10,10)

    def paint(self, painter, option, index):
        groupingActive = self.parent().isGroupigActive()
        indexGroupActive = self.parent().model().getGroupingStateByTableIndex(index)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        if groupingActive:
            centerPoint = option.rect.center()
            background = QRect(option.rect)

            b  = QBrush(Qt.BrushStyle.NoBrush)
            r = 5
            if self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
                b = QBrush(QColor(HOVER_COLOR)) 
                r = 7
            elif self.parent().selectionModel().isSelected(self.parent().model().index(index.row(),0)):
                b =  QBrush(QColor("lightgrey"))

            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(background)
            
            if indexGroupActive:
                pen = QPen(QColor("darkgrey"))
                groupColor = self.parent().model().getGroupColorByTableIndex(index)
                brush = QBrush(QColor(groupColor))
                painter.setBrush(brush)
            else:
                pen = QPen(QColor("darkgrey" if self.parent().focusColumn != self.highLightColumn else "#B84D29"))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            #set pen width
            pen.setWidthF(0.75)
            painter.setPen(pen)

            painter.drawEllipse(QPointF(centerPoint),r,r)

        elif self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            centerPoint = option.rect.center()
            background = QRect(option.rect)
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(background)

            pen = QPen(QColor("darkgrey" if self.parent().focusColumn != self.highLightColumn else "#B84D29"))
            pen.setWidthF(0.75)
            painter.setPen(pen)
            
            painter.drawEllipse(QPointF(centerPoint),5,5)
        
        elif self.parent().selectionModel().isSelected(self.parent().model().index(index.row(),0)):
           b = QBrush(QColor("lightgrey"))
           painter.setBrush(b)
           painter.setPen(Qt.PenStyle.NoPen)
           painter.drawRect(option.rect)


class MaskDelegate(QStyledItemDelegate):
    def __init__(self,parent, highLightColumn = 1):
        super(MaskDelegate,self).__init__(parent)
        self.highLightColumn = highLightColumn
    
    def paint(self, painter, option, index):
        
        if self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            centerPoint = option.rect.center()
            rect = option.rect
            background = QRect(option.rect)


            x0 = centerPoint.x()
            y0 = centerPoint.y()
            h = rect.height()/3
            w = h #square paint area


            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(background)
            
            
            # draw left (lighter) half
            path = QPainterPath()
            path.moveTo(x0-w,y0-h/1.5)
            path.quadTo(x0-w,y0-h,x0,y0-h)
            path.lineTo(x0,y0+h)
            path.quadTo(x0-w/2,y0+h,x0-w,y0)
            path.lineTo(x0-w,y0-h/1.5)
            b = QBrush(QColor("lightgrey" if self.parent().focusColumn != self.highLightColumn  else "#B84D29"))
            painter.setBrush(b)
            painter.drawPath(path)

            #draw right (Dark) half
            path.clear()
            path.moveTo(x0+w,y0-h/1.5)
            path.quadTo(x0+w,y0-h,x0,y0-h)
            path.lineTo(x0,y0+h)
            path.quadTo(x0+w/2,y0+h,x0+w,y0)
            path.lineTo(x0+w,y0-h/1.5)
            
            b = QBrush(QColor("darkgrey" if self.parent().focusColumn != self.highLightColumn  else "#824329"))
            painter.setBrush(b)
            painter.drawPath(path)

            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)

            #draw smile
            path.clear()
            path.moveTo(x0-w/2,y0+h/3)
            path.quadTo(x0,y0+h/2,x0+w/2,y0+h/3)
            path.quadTo(x0,y0+h,x0-w/2,y0+h/3)
            painter.drawPath(path)
            

            #draw left eye
            path.clear()
            eyeHeight = h/2
            eyeWidth = h/2
            eyeLeftRect = QRectF(x0-w/4-h/2,y0-h/2,eyeWidth,eyeHeight)
            path.moveTo(eyeLeftRect.center())
            path.arcTo(eyeLeftRect,0,180)
  
            painter.drawPath(path)
 
            #darw right eye
            path.clear()
            eyeRightRect = QRectF(x0+w/4,y0-h/2,eyeWidth,eyeHeight)
            path.moveTo(eyeRightRect.center())
            path.arcTo(eyeRightRect,0,180)
            painter.drawPath(path)
        
        elif self.parent().selectionModel().isSelected(index):
               b = QBrush(QColor("lightgrey"))
               painter.setBrush(b)
               painter.setPen(Qt.PenStyle.NoPen)
               painter.drawRect(option.rect)


class FilterDelegate(QStyledItemDelegate):
    def __init__(self,parent,highLightColumn = 2):
        super(FilterDelegate,self).__init__(parent)
        self.highLightColumn = highLightColumn

    def sizeHint(self):
        ""
        return QSize(10,10)

    def paint(self, painter, option, index):
        
        if self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            centerPoint = option.rect.center()
            background = QRect(option.rect)
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(background)            
            rect = option.rect
            x0 = centerPoint.x()
            y0 = centerPoint.y()
            h = rect.height()/4
            w = rect.width()
            path = QPainterPath()
            painter.setPen(Qt.PenStyle.NoPen)

            path.moveTo(x0-w/5, y0-h)
            path.lineTo(x0+w/5,y0-h)
            path.lineTo(x0+w/10,y0)
            path.lineTo(x0+w/10,y0+h)

            path.lineTo(x0-w/10,y0+h)
            path.lineTo(x0-w/10,y0)
            path.lineTo(x0-w/5,y0-h)
            b = QBrush(QColor("darkgrey" if self.parent().focusColumn != self.highLightColumn else "#0073B0"))
            painter.setBrush(b)
            painter.drawPath(path)

            #draw ellipse for 3d effect
            brush = QBrush(QColor("lightgrey"))
            painter.setBrush(brush)
            pen = QPen(QColor("black"))
            pen.setWidthF(0.1)
            painter.setPen(pen)

            painter.drawEllipse(QRectF(x0-w/5,y0-h-h/3,2*w/5,h/1.5))
        
        elif self.parent().selectionModel().isSelected(self.parent().model().index(index.row(),0)):

               b = QBrush(QColor("lightgrey"))
               painter.setBrush(b)
               painter.setPen(Qt.PenStyle.NoPen)
               painter.drawRect(option.rect)
                        

class ItemDelegate(QStyledItemDelegate):
    def __init__(self,parent, highLightColumn=0):
        super(ItemDelegate,self).__init__(parent)
        self.highLightColumn = highLightColumn

    def paint(self, painter, option, index):

        painter.setFont(getStandardFont())
        rect = option.rect
        if self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(option.rect)
            self.addText(index,painter,rect)

        elif self.parent().selectionModel().isSelected(index):
            b = QBrush(QColor("lightgrey"))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(option.rect)
            self.addText(index,painter,rect)
        
        else:
            self.addText(index,painter,rect)
    
    def addText(self,index,painter,rect):
        ""
        painter.setPen(QPen(QColor("black")))
        rect.adjust(9,0,0,0)
        painter.drawText(rect,   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.parent().model().data(index,Qt.ItemDataRole.DisplayRole))
       
    def setEditorData(self,editor,index):
        editor.setFont(getStandardFont())
        editor.setAutoFillBackground(True)
        editor.setText(self.parent().model().data(index,Qt.ItemDataRole.DisplayRole))





class AddDelegate(QStyledItemDelegate):
    def __init__(self,parent,highLightColumn = 1):
        super(AddDelegate,self).__init__(parent)
        self.highLightColumn = highLightColumn

    def paint(self, painter, option, index):
        if self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(option.rect)
            pen = QPen(QColor("darkgrey" if self.parent().focusColumn != self.highLightColumn else "#288C36"))
            pen.setWidthF(3)
            painter.setPen(pen)
            centerPoint = option.rect.center() 
            h = option.rect.height() / 2

            painter.drawLine(QPointF(centerPoint.x()-h/3, centerPoint.y() - 0.5), QPointF(centerPoint.x()+h/3,centerPoint.y()-0.5))
            try:
                if not self.parent().model().getColumnStateByTableIndex(index):
                    painter.drawLine(QPointF(centerPoint.x(), centerPoint.y()-h/3), QPointF(centerPoint.x(),centerPoint.y()+h/3))
            except Exception as e:
                print(e)
        
        elif self.parent().selectionModel().isSelected(self.parent().model().index(index.row(),0)):
               b = QBrush(QColor("lightgrey"))
               painter.setBrush(b)
               painter.setPen(Qt.PenStyle.NoPen)
               painter.drawRect(option.rect)
                


class CopyDelegate(QStyledItemDelegate):
    def __init__(self,parent,highLightColumn=1):
        super(CopyDelegate,self).__init__(parent)
        self.highLightColumn = highLightColumn

    def paint(self, painter, option, index):
        
        if self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            rect = option.rect
            background = QRect(rect)
            #rect.adjust(8,0,-8,0)
            centerPoint = rect.center()
            

            x0 = centerPoint.x()
            y0 = centerPoint.y()
            h = rect.height()/8
            w = rect.width()
            path = QPainterPath()

            path.moveTo(x0-h,y0-0.8*h)
            path.lineTo(x0+h,y0-0.8*h)
            path.lineTo(x0+h,y0-2*h)

            path.lineTo(x0+3*h,y0)

            path.lineTo(x0+h,y0+2*h)
            path.lineTo(x0+h,y0+0.8*h)
            path.lineTo(x0-h,y0+0.8*h)
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(background)
            pen = QPen(QColor("black"))
            pen.setWidthF(0.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            b = QBrush(QColor("darkgrey" if self.parent().focusColumn != self.highLightColumn  else "#B84D29"))
            painter.setBrush(b)
            painter.drawPath(path)
        
        elif self.parent().selectionModel().isSelected(self.parent().model().index(index.row(),0)):
               b = QBrush(QColor("lightgrey"))
               painter.setBrush(b)
               painter.setPen(Qt.PenStyle.NoPen)
               painter.drawRect(option.rect)            
  


class DelegateColor(QStyledItemDelegate):
    def __init__(self,parent):
        super(DelegateColor, self).__init__(parent)

    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)
       # print(self.parent().focusRow)
        
        if self.parent().model().getCheckStateByTableIndex(index):
            painter.save()
            color = self.parent().model().getColor(index)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            brush = QBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(brush)
            r = QRect(option.rect)
            r.adjust(8, 8, -8, -8)
            painter.drawRoundedRect(r,4,4)
            painter.restore()      

        elif self.parent().focusRow is not None and index.row() == self.parent().focusRow and self.parent().focusColumn is not None:
            
            pen = painter.pen()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
            brush = QBrush(QColor("lightgrey"))
            pen.setColor(Qt.transparent)
            painter.setPen(pen)
            painter.setBrush(brush)
            r = QRect(option.rect)
            r.adjust(8, 8, -8, -8)
            painter.drawRoundedRect(r,4,4)

