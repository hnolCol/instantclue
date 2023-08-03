
from PyQt5.QtCore import *
from PyQt5.QtCore import QModelIndex
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
import pandas as pd 
import numpy as np 
import os 

from ...utils import getHoverColor, getStdTextColor, getStandardFont


class ItemDelegate(QStyledItemDelegate):
    def __init__(self,parent):
        super(ItemDelegate,self).__init__(parent)
    
    def paint(self, painter, option, index):
        painter.setFont(getStandardFont())
        rect = option.rect
        if self.parent().mouseOverItem is not None and index.row() == self.parent().mouseOverItem:
            b = QBrush(QColor(getHoverColor()))
            painter.setBrush(b)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(option.rect)
            self.addText(index,painter,rect)
        
        else:
            self.addText(index,painter,rect)
    
    def addText(self,index,painter,rect):
        ""
        painter.setPen(QPen(QColor(getStdTextColor())))
        rect.adjust(9,0,0,0)
        painter.drawText(rect,   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.parent().model().data(index,Qt.ItemDataRole.DisplayRole))
       
    def setEditorData(self,editor,index):
        editor.setFont(getStandardFont())
        editor.setAutoFillBackground(True)
        editor.setText(self.parent().model().data(index,Qt.ItemDataRole.DisplayRole))


class ICModelBase(QAbstractTableModel):
    def __init__(self,*args,**kwargs):
        super(ICModelBase,self).__init__(*args,**kwargs)
        self._labels = pd.DataFrame()
    
    def rowCount(self, parent=QModelIndex()) -> int:
        ""
        return self._labels.index.size

    def columnCount(self, parent: QModelIndex = ...) -> int:
        ""
        return 0
    
    def completeDataChanged(self) -> None:
        ""
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))

    def dataAvailable(self) -> bool:
        ""
        return self._labels.index.size > 0 
    
    def getCurrentGroup(self) -> str:
        ""
        if hasattr(self.parent(),"mouseOverItem"):
            mouseOverItem = self.parent().mouseOverItem
     
            if mouseOverItem is not None:
            
                return self._labels.iloc[mouseOverItem].loc["group"]

    def getCurrentInternalID(self) -> str:
        ""
        if hasattr(self.parent(),"mouseOverItem"):
            mouseOverItem = self.parent().mouseOverItem
            if mouseOverItem is not None and "internalID" in self._labels.columns: 
                return self._labels.iloc[mouseOverItem].loc["internalID"]

    def getDataIndex(self,row) -> np.ndarray:
        ""
        if self.validDataIndex(row):
            return self._labels.index[row]
        
    def getSelectedData(self,indexList) -> pd.Series:
        ""
        dataIndices = [self.getDataIndex(tableIndex.row()) for tableIndex in indexList]
        return self._labels.loc[dataIndices]

    def getLabels(self) -> pd.Series:
        ""
        return self._labels
        
    def getLabelsName(self):
        ""
        return self._labels.columns
    
    def validDataIndex(self,row) -> bool:
        ""
        return row <= self._labels.index.size - 1

    def rowRangeChange(self,row1, row2) -> None:
        ""
        self.dataChanged.emit(self.index(row1,0),self.index(row2,self.columnCount()-1))

    def rowDataChanged(self, row) -> None:
        ""
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()-1))

    def setNewData(self,labels : pd.Series) -> None:
        ""
        self.initData(labels)

    def resetView(self) -> None:
        ""
        self._labels = pd.Series(dtype="object")
        self._inputLabels = self._labels.copy()

class ICTableBase(QWidget):
    selectionChanged = pyqtSignal()

    def __init__(self, mainController, *args,**kwargs):

        super(ICTableBase,self).__init__(*args,**kwargs)
        self.mC = mainController
        self.title = ""
        self.encodedColumnNames = []

        self.setMaximumHeight(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)

    def setTitle(self,title):
        ""
        if isinstance(title,str):
            self.titleLabel.setText(title)
            self.title = title

    def setEncodedColumnNames(self,encoedColumnNames : pd.Series):
        ""
        self.encodedColumnNames = encoedColumnNames
    
    def setData(self,data, title=None, isEditable = True, encodedColumnNames = None):
        "" 
        if isinstance(data,pd.DataFrame):
            self.setTitle(title)
            self.setEncodedColumnNames(encodedColumnNames)
            self.table.model().layoutAboutToBeChanged.emit()
            self.table.model().isEditable = isEditable
            self.table.createMenu()
            self.table.model().initData(data)
            self.table.model().layoutChanged.emit()
            self.table.model().completeDataChanged()
            self.setWidgetHeight()
            
    def setWidgetHeight(self):
        ""
        if "\n" in self.title:
            linesTakenByTitle = self.title.count("\n")
        else:
            linesTakenByTitle = 1 
        
        rowCount = self.table.model().rowCount()
        if rowCount == 0:
            maxHeight = 0
        else:
            maxHeight = int(rowCount * (self.table.rowHeight+2) + linesTakenByTitle * 25 + 55) #header + title
     
        self.setMaximumHeight(maxHeight)
    
    def reset(self):
        ""
        self.table.model().layoutAboutToBeChanged.emit()
        self.table.model().resetView()
        self.table.model().layoutChanged.emit()
        self.table.model().completeDataChanged()
        self.setWidgetHeight()


    def subsetSelection(self):
        ""
        rowIndex = self.table.rightClickedRowIndex
        internalID = self.table.model().getInternalIDByRowIndex(rowIndex)
        groupName = self.table.model().getGroupByInternalID(internalID)
        exists, graph =  self.mC.getGraph()
        if exists and groupName is not None:
            graph.subsetDataOnInternalID(internalID,groupName)
           
    def leaveEvent(self, event = None):
        ""
        if hasattr(self,"table"):
            self.table.mouseOverItem = None
        if hasattr(self,"highlightColorGroup"):
            self.highlightColorGroup(reset=True)

    def saveModelDataToExcel(self):
        "Allows export to excel file."
        baseFilePath = os.path.join(self.mC.config.getParam("WorkingDirectory"),"ICExport")
        fname,_ = QFileDialog.getSaveFileName(self, 'Save file', baseFilePath,
                        "Excel files (*.xlsx)")
                #if user cancels file selection, return function
        if fname:
            self.table.model()._labels.to_excel(fname,sheet_name="ICExport")
    
    def copyToClipboard(self):
        "Pastes model data to clipboard."
        self.table.model()._labels.to_clipboard()