import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from ..utils import createLabel, createTitleLabel, getHoverColor
import pandas as pd
import numpy as np


class ResortableTable(QDialog):

    def __init__(self,inputLabels, deleteItems = False, *args,**kwargs):
        super(ResortableTable,self).__init__(*args, **kwargs)
        self.deleteItems = deleteItems
        self.savedData = None
        if self.prepareData(inputLabels):
            self.__controls()
            self.__layout()
            self.__connectEvents()

    def __controls(self):
        ""
        self.titleLabel = createTitleLabel("Use Drag & Drop to sort items.",fontSize=15)
        self.infoLabel = createLabel("Use DEL or backspace to delete a selection. Graph will update only after closing this dialog window.")
        self.table = ResortTableWidget(parent = self )
        self.model = ResortTableModel(parent=self.table, inputLabels = self.inputLabels)
        self.table.setModel(self.model)

        self.saveButton = QPushButton("Save")
        self.cancelButton = QPushButton("Cancel")


    def __layout(self):
        ""

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.infoLabel)
        self.layout().addWidget(self.table)

        hbox = QHBoxLayout()
        hbox.addWidget(self.saveButton)
        hbox.addWidget(self.cancelButton)

        self.layout().addLayout(hbox)

    def __connectEvents(self):

        self.saveButton.clicked.connect(self.saveData)
        self.cancelButton.clicked.connect(self.close)

    def prepareData(self,inputLabels):
        ""
        if not isinstance(inputLabels,pd.Series):
            try:
                inputLabels = pd.Series(inputLabels)
            except:
                print("Could not build pandas Series from input.")
                return False

        self.inputLabels = inputLabels

        return not self.inputLabels.size == 0

    def saveData(self,event=None):
        ""
        self.savedData = self.model._labels
        self.accept()
        self.close()
   
class ResortTableWidget(QTableView):
    def __init__(self, menu = None, *args,**kwargs):
        super(ResortTableWidget,self).__init__(*args,**kwargs)
        self.rightClick = False
        self.menu = menu
        #set drag and drop params
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.verticalHeader().setDefaultSectionSize(15)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        #self.setDragDropMode(QAbstractItemView.InternalMove)

        p = self.palette()
        p.setColor(QPalette.ColorRole.Highlight,QColor(getHoverColor()))
        p.setColor(QPalette.ColorRole.HighlightedText  ,QColor("black"))
        self.setPalette(p)
       
    
    def dragEvent(self,event):
        ""
        event.accept()

    def dragMoveEvent(self, event):
        event.accept()
        self.hideDragLabels(event)
        
    def dropEvent(self,event):
        ""
        event.accept()
        self.resetDragEvent()
        self.model().completeDataChanged()

    def resetDragEvent(self):
        ""
        del self.draggedRows
        del self.dragStartIndex
        #self.model().resetDragEvent()

    def mouseEventToIndex(self,event):
        "Converts mouse event on table to tableIndex"
        row = self.rowAt(int(event.pos().y()))
        column = self.columnAt(int(event.pos().x()))
        return self.model().index(row,column)

    def mouseReleaseEvent(self,event):
        ""
        if self.rightClick:
            if self.menu is not None:
                pos = self.mapToGlobal(event.pos())
                self.menu.exec(QPoint(int(pos.x()),int(pos.y())))

    def mousePressEvent(self,event):
        ""
        if event.buttons() == Qt.MouseButton.RightButton:
            self.rightClick = True
        elif event.buttons() == Qt.MouseButton.LeftButton:
            super(QTableView,self).mousePressEvent(event)
            self.rightClick = False
        else:
            self.rightClick = False

    def hideDragLabels(self,event):
        
        if not hasattr(self,"dragStartIndex"):
            self.dragStartIndex = self.mouseEventToIndex(event)
        if not hasattr(self,"draggedRows"):
            self.draggedRows = self.selectionModel().selectedRows()
            self.model().setDraggedIndicies(self.draggedRows)
        currentIndex = self.mouseEventToIndex(event) 
        if self.model().onlyDragNoResort:
            rowOffset = 0
        else:
            if currentIndex.row() == -1:
                rowOffset = self.model().rowOffset + 1
            else:
                rowOffset = currentIndex.row() - self.dragStartIndex.row()
       
        if self.model().setRowOffset(rowOffset):
            self.model().setHiddenIdx(self.draggedRows)
            self.moveSelection()
            self.model().resortData()
            self.model().completeDataChanged()
       
    def moveSelection(self):
        self.selectionModel().clear()
        for idx in self.model().hiddenIndex:
            tableIndex = self.model().index(idx,0)
            self.selectionModel().select(tableIndex,QItemSelectionModel.SelectionFlag.Select)
    
    def keyPressEvent(self,e):
        ""
        if e.key() in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:
            idx = self.getSelectedRows() 
            self.model().deleteRowsByTableIndex(idx)

    def getSelectedRows(self):
        ""
        return self.selectionModel().selectedRows()


class ResortTableModel(QAbstractTableModel):
    
    def __init__(self, parent=None, inputLabels = pd.Series(dtype="object"), title = "Data"):
        super(ResortTableModel, self).__init__(parent)
        self.initData(inputLabels)
        self.title = title
        self.hiddenIndex = []
        self.rowOffset = 0
        self.onlyDragNoResort = False

    def initData(self,inputLabels):
        ""
        #reference to input
        self._inputLabels = inputLabels.astype("object")
        self.hiddenLabels = pd.Series(dtype="object") 
        self._labels = self._inputLabels.copy()
        self._dragLabels = self._inputLabels.copy()

    def rowCount(self, parent=QModelIndex()):
        return self._labels.index.size
    
    def getFont(self):
        ""
        font = QFont()
        font.setFamily("Arial")
        font.setWeight(300)
        font.setPointSize(9)
        return font
    
    def columnCount(self, parent=QModelIndex()):
        
        return 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
  
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.ItemDataRole.FontRole:
            return self.getFont()
        elif role == Qt.ItemDataRole.DisplayRole:
            try:
                return str(self._labels.iloc[index.row()])
            except:
                return ""
        elif role == Qt.ItemDataRole.BackgroundRole:
            return QBrush(QColor("white"))#TABLE_ODD_ROW_COLOR
            
    def setData(self,index,value,role):
        ""
        if role == Qt.ItemDataRole.EditRole:
    
            return True
    
    def deleteRowsByTableIndex(self,tableIndexes):
        "Remove rows by table index"
        tableIdxs = [tidx.row() for tidx in tableIndexes]
        boolIdx = np.array([idx in tableIdxs for idx,_ in enumerate(self._labels.index)])
        self.layoutAboutToBeChanged.emit()
        self.initData(self._labels.loc[~boolIdx])
        self.layoutChanged.emit()

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self.title)
        elif orientation == Qt.Orientation.Vertical and role == Qt.ItemDataRole.DisplayRole:
            return str(self._labels.index[col])
        elif role == Qt.ItemDataRole.BackgroundRole:
            if orientation == Qt.Orientation.Horizontal:
                return QBrush(QColor("#4C626F"))
            else:
                return QBrush(QColor("grey"))
        elif role == Qt.ItemDataRole.ForegroundRole:
            return QBrush(QColor("black"))
        return None

    def flags(self, index):
        ""
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled 

    def completeDataChanged(self):
        ""
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))

    def setHiddenIdx(self,indexList):
        ""
        self.hiddenIndex = [idx.row() + self.rowOffset for idx in indexList]

    def setRowOffset(self,rowOffset):
        ""
        self.saveOffset = self.rowOffset
        self.rowOffset = rowOffset
        return not self.saveOffset == rowOffset

    def resetDragEvent(self):
        ""
        self._dragLabels = self._labels.copy()
        self.rowOffset = 0
        self.saveOffset = 0
        self.hiddenIndex = []

    def setDraggedIndicies(self,draggedRows):
        ""
        self.draggedRows = [idx.row() for idx in draggedRows]
        self.draggedIndices = self._labels.iloc[self.draggedRows].index

    def getDraggedlabels(self):
        ""
        if hasattr(self,"draggedIndices"):
            return self._labels.loc[self.draggedIndices]

    def hideLabels(self, labelsToHide):
        ""
        diffLabels = labelsToHide.loc[labelsToHide.index.difference(self.hiddenLabels.index)]

        self.hiddenLabels = pd.concat([self.hiddenLabels.copy(),diffLabels], ignore_index=True)

        self._labels = self._inputLabels.loc[~self._inputLabels.index.isin(self.hiddenLabels.index)]
        self._dragLabels = self._labels.copy() 

    def showHiddenLabels(self,labelsToShow):
        "Hide Labels"
        labelsHidden = self.hiddenLabels.index.isin(labelsToShow.index)
        self.hiddenLabels = self.hiddenLabels.loc[~labelsHidden]
        self._labels = self._inputLabels.loc[~self._inputLabels.index.isin(self.hiddenLabels.index)]
        self._dragLabels = self._labels.copy() 

    def resortData(self):
        ""
        if self.onlyDragNoResort:
            return
        if self.saveOffset == self.rowOffset:
            return
        else:
            selectedRows = self.hiddenIndex 
            if len(selectedRows) == 0:
                return
            firstRow = selectedRows[0]
            
            boolIdx = self._dragLabels.index.isin(self.draggedIndices.values) == False
            idxRaw = self._dragLabels.index.values[boolIdx].tolist()

            if firstRow == 0:
                idxBefore = []
            else:
                idxBefore = idxRaw[:firstRow]
            idxAfter = idxRaw[firstRow:]
          
            idxDragged = self.draggedIndices.values.tolist() 
            idx = pd.Series(idxBefore + idxDragged + idxAfter)
            self._labels = self._dragLabels.loc[idx]
            self._dragLabels = self._dragLabels.loc[idx]
            

