from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QStyledItemDelegate, QSpinBox
import numpy as np

class SpinBoxDelegate(QStyledItemDelegate):
    
    def __init__(self,*args,**kwargs):
        super(SpinBoxDelegate, self).__init__(*args,**kwargs)
    
    def paint(self, painter, option, index):
        ""
        QStyledItemDelegate.paint(self, painter, option, index)
        r = option.rect.height()/2.5
        maxSize = self.parent().model().maxSize
        minSize = self.parent().model().minSize
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        if minSize == maxSize:
            scaledR = r 
        else:
            scaledR = ((np.sqrt(value) - minSize) / (maxSize - minSize) * (0.95 - 0.4) + 0.4) * r
        
        pen = painter.pen()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        pen.setColor(QColor("darkgrey"))
        pen.setWidthF(0.5)
        painter.setPen(pen)    
        brush = QBrush(QColor("#efefef"))  
        painter.setBrush(brush)     
        centerPoint = option.rect.center()
        painter.drawEllipse(QPointF(centerPoint),scaledR ,scaledR )

    def createEditor(self, parent, option, index):
        "Create Spinbox Editor"
        editor = QSpinBox(parent)
        editor.setFrame(False)
        #min and max should be define on init.
        editor.setMinimum(2)
        editor.setMaximum(1000)
        return editor

    def setEditorData(self, spinBox, index):
        "Set the init data to the spinbox"
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        spinBox.setValue(int(float(value)))

    def setModelData(self, spinBox, model, index):
        ""
        spinBox.interpretText()
        value = spinBox.value()
        model.setData(index, value, Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        ""
        editor.setGeometry(option.rect)