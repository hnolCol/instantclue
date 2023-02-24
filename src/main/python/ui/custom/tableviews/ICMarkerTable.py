from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ..utils import clearLayout, getStandardFont
from ...utils import HOVER_COLOR, createSubMenu, createMenu, createLabel, createTitleLabel
from .ICColorTable import ICColorSizeTableBase

from ...delegates.quickSelectDelegates import DelegateColor #borrow delegate

import pandas as pd
import numpy as np
import os


class ICMarkerTable(ICColorSizeTableBase):
   ## clorMapChanged = pyqtSignal() 
    def __init__(self, *args,**kwargs):

        super(ICMarkerTable,self).__init__(*args,**kwargs)
        
        self.selectionChanged.connect(self.updateMarkerInGraph)
      #  self.clorMapChanged.connect(self.updateColorsByColorMap)
        self.__controls()
        self.__layout()
        

    def __controls(self):
        ""
        self.mainHeader = createTitleLabel("Markers",fontSize = 14)
        self.mainHeader.setWordWrap(True)
        self.titleLabel = createLabel(text = self.title)
        self.table = MarkerTable(parent = self, mainController=self.mC)
        self.model = MarkerTableModel(parent=self.table)
        self.table.setModel(self.model)

        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch) 
        self.table.resizeColumns()
       # self.table.setItemDelegateForColumn(0,DelegateColor(self.table))

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mainHeader)
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.table)  

    # @pyqtSlot()
    # def updateColorsByColorMap(self):
    #     ""
        
    #     if self.model.rowCount() > 0:
    #         self.table.createMenu()
    #         if self.mC.getPlotType() == "scatter":
    #             funcProps = {"key":"plotter:getScatterColorGroups","kwargs":{"dataID":self.mC.getDataID(),
    #                 "colorColumn":None,
    #                 "colorColumnType":None,
    #                 "colorGroupData":self.model._labels}}
            
    #             self.mC.sendRequestToThread(funcProps)
    #         else:
    #             colorList = self.mC.colorManager.getNColorsByCurrentColorMap(N = self.model.rowCount())
    #             self.model.updateColors(colorList)
    
    def updateMarkerInGraph(self):
        ""
        exists, graph =  self.mC.getGraph()
        try:
            if exists:
                graph.setHoverObjectsInvisible()
                graph.updateGroupColors(self.table.model().getLabels(),self.table.model().getItemChangedInternalID())
        except Exception as e:
            print(e) 

    def addLegendToGraph(self, ignoreNaN = False, legendKwargs = {}):
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.addMarkerLegend(self.model._labels,title = self.title, legendKwargs = legendKwargs)
            self.model.completeDataChanged()

    def removeFromGraph(self):
        ""
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.setLegendInvisible()
            graph.setNaNColor()
            self.model.completeDataChanged()
       

class MarkerTableModel(QAbstractTableModel):
    
    def __init__(self, labels = pd.DataFrame(), parent=None, isEditable = False):
        super(MarkerTableModel, self).__init__(parent)
        self.initData(labels)
        self.isEditable = isEditable

    def initData(self,labels):

        self._labels = labels
        self._inputLabels = labels.copy()
        
        self.columnInGraph = pd.Series(np.zeros(shape=labels.index.size), index=labels.index)
        self.setDefaultSize()
        self.lastSearchType = None
        

    def rowCount(self, parent=QModelIndex()):
        
        return self._labels.index.size

    def columnCount(self, parent=QModelIndex()):
        
        return 2
    
    def dataAvailable(self):
        ""
        return self._labels.index.size > 0 

    def setDefaultSize(self,size=50):
        ""
        self.defaultSize = size

    def getDataIndex(self,row):
        ""
        if self.validDataIndex(row):
            return self._labels.index[row]
        
    def getColumnStateByDataIndex(self,dataIndex):
        ""
        return self.columnInGraph.loc[dataIndex] == 1

    def getColumnStateByTableIndex(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            return self.getColumnStateByDataIndex(dataIndex)

    def setColumnState(self,tableIndex, newState = None):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex is not None:
            if newState is None:
                newState = not self.columnInGraph.loc[dataIndex]
            self.columnInGraph.loc[dataIndex] = newState
            return newState
    
    def setColumnStateByData(self,columnNames,newState):
        ""
        idx = self._labels[self._labels.isin(columnNames)].index
        if not idx.empty:
            self.columnInGraph[idx] = newState

    def updateData(self,value,index):
        ""
        dataIndex = self.getDataIndex(index.row())
        if dataIndex is not None:
            self._labels[dataIndex] = value
            self._inputLabels = self._labels.copy()

    def validDataIndex(self,row):
        ""
        return row <= self._labels.index.size - 1

    def deleteEntriesByIndexList(self,indexList):
        ""
        dataIndices = [self.getDataIndex(tableIndex.row()) for tableIndex in indexList]
        self._labels = self._labels.drop(dataIndices)
        self._inputLabels = self._labels.copy()
        self.completeDataChanged()

    def deleteEntry(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex in self._inputLabels.index:
            self._labels = self._labels.drop(dataIndex)
            self._inputLabels = self._labels
            self.completeDataChanged()

    def getLabels(self):
        ""
        return self._labels

    def getSelectedData(self,indexList):
        ""
        dataIndices = [self.getDataIndex(tableIndex.row()) for tableIndex in indexList]
        return self._labels.loc[dataIndices]

    def getCurrentGroup(self):
        ""
        mouseOverItem = self.parent().mouseOverItem
     
        if mouseOverItem is not None:
            
            return self._labels.iloc[mouseOverItem].loc["group"]

    def getCurrentInternalID(self):
        ""
        mouseOverItem = self.parent().mouseOverItem
        if mouseOverItem is not None and "internalID" in self._labels.columns: 
            return self._labels.iloc[mouseOverItem].loc["internalID"]

    def getGroupByInternalID(self,internalID):
        ""
        if "internalID" in self._labels.columns:
            boolIdx = self._labels["internalID"] == internalID
            if np.any(boolIdx):
                return self._labels.loc[boolIdx,"group"].values[0]

    def getInternalIDByRowIndex(self,row):
        ""
        return self._labels.iloc[row].loc["internalID"]

    def getItemChangedInternalID(self):
        ""
        if self.parent().colorChangedForItem is not None and "internalID" in self._labels.columns:
            return self._labels.iloc[self.parent().colorChangedForItem].loc["internalID"]
            
    def setData(self,index,value,role):
        ""
        row =index.row()
        indexBottomRight = self.index(row,self.columnCount())
        if role == Qt.ItemDataRole.UserRole:
            self.dataChanged.emit(index,indexBottomRight)
            return True
        if role == Qt.CheckStateRole:
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True
        elif role == Qt.EditRole:
            if not self.isEditable:
                return False
            if index.column() != 0:
                return False
            newValue = str(value)
            oldValue = str(self._labels.iloc[index.row()])
            columnNameMapper = {oldValue:newValue}
            if oldValue != newValue:
                self.parent().renameColumn(columnNameMapper)
                self.updateData(value,index)
                self.dataChanged.emit(index,index)
            return True
        

    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
        
        if not index.isValid(): 

            return QVariant()
            
        elif role == Qt.ItemDataRole.DisplayRole: 
            return str(self._labels.iloc[index.row(),index.column()])
        
        elif role == Qt.ItemDataRole.FontRole:

            return getStandardFont()

        elif role == Qt.ItemDataRole.ToolTipRole:

            return "Markers are from matplotlib package."

        elif self.parent().mouseOverItem is not None and role == Qt.ItemDataRole.BackgroundRole and index.row() == self.parent().mouseOverItem:
            return QColor(HOVER_COLOR)
            
    def flags(self, index):
        "Set Flags of Column"
        if index.column() == 0:
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled #| Qt.ItemIsEditable
        else:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def setNewData(self,labels):
        ""
        self.initData(labels)
        self.completeDataChanged()
    
    def completeDataChanged(self):
        ""
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))

    def rowRangeChange(self,row1, row2):
        ""
        self.dataChanged.emit(self.index(row1,0),self.index(row2,self.columnCount()-1))

    def rowDataChanged(self, row):
        ""
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()-1))

    def resetView(self):
        ""
        self._labels = pd.Series(dtype="object")
        self._inputLabels = self._labels.copy()
        self.completeDataChanged()




class MarkerTable(QTableView):

    def __init__(self, parent=None, rowHeight = 22, mainController = None):

        super(MarkerTable, self).__init__(parent)
       
        self.setMouseTracking(True)
        self.setShowGrid(True)
        self.verticalHeader().setDefaultSectionSize(rowHeight)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)

        self.mC = mainController
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) 
        

        self.createMenu()

        self.rowHeight      =   rowHeight
        self.rightClick     =   False
        self.mouseOverItem  =   None
        self.colorChangedForItem = None
        
        p = self.palette()
        p.setColor(QPalette.ColorRole.Highlight,QColor(HOVER_COLOR))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor("black"))
        self.setPalette(p)

        self.setStyleSheet("""QTableView {background-color: #F6F6F6;border:None};""")

    def colorChangedFromMenu(self,event=None, hexColor = ""):
        ""
        rowIndex = self.rightClickedRowIndex
        
        dataIndex = self.model().getDataIndex(rowIndex)
        self.model().setColor(dataIndex,hexColor)
        self.parent().selectionChanged.emit()
        self.model().rowDataChanged(rowIndex)

    def createMenu(self):
        ""
        legendLocations = ["upper right","upper left","center left","center right","lower left","lower right"]
        menu = createSubMenu(None,["Subset by ..","Add Legend at .."])
        menu["main"].addAction("Remove", self.parent().removeFromGraph)
        #menu["main"].addAction("Add to graph", self.parent().addLegendToGraph)
        menu["main"].addAction("Save to xlsx",self.parent().saveModelDataToExcel)
        menu["Subset by .."].addAction("Group", self.parent().subsetSelection)

        for legendLoc in legendLocations:
            menu["Add Legend at .."].addAction(legendLoc,lambda lloc = legendLoc: self.parent().addLegendToGraph(legendKwargs = {"loc":lloc}))
        self.menu = menu["main"]
    
    def leaveEvent(self,event=None):
        ""
        if hasattr(self, "mouseOverItem") and self.mouseOverItem is not None:
            prevMouseOver = int(self.mouseOverItem)
            self.mouseOverItem = None
            self.model().rowDataChanged(prevMouseOver)

    def mouseEventToIndex(self,event):
        "Converts mouse event on table to tableIndex"
        row = self.rowAt(event.pos().y())
        column = self.columnAt(event.pos().x())
        return self.model().index(row,column)
    
    def mousePressEvent(self,e):
        ""
       # super().mousePressEvent(e)
        if e.buttons() == Qt.RightButton:
            self.rightClick = True
        else:
            self.rightClick = False
    
    def mouseReleaseEvent(self,e):
        ""
        #reset move signal
        "Handles Mouse Release Events"
        if not self.model().dataAvailable():
            return
       
        try:
            tableIndex = self.mouseEventToIndex(e)
            if tableIndex is None:
                return
            tableIndexCol = tableIndex.column()
            if tableIndexCol == 0 and self.model().isEditable:
                dataIndex = self.model().getDataIndex(tableIndex.row())
                
                self.parent().selectionChanged.emit()
                self.model().rowDataChanged(tableIndex.row())

                
            elif tableIndexCol == 1 and self.rightClick:
               # idx = self.model().index(0,0)
                self.rightClickedRowIndex = tableIndex.row()
                self.menu.exec(QCursor.pos() + QPoint(4,4))
                
                self.clickedRow = None 

        except Exception as e:
            print(e)


    def mouseMoveEvent(self,event):
        
        ""
        if not self.model().dataAvailable():
            return
        rowAtEvent = self.rowAt(event.pos().y())
        if rowAtEvent == -1:
            self.mouseOverItem = None
        else:
            self.mouseOverItem = rowAtEvent
        self.model().rowDataChanged(rowAtEvent)
 
    def resizeColumns(self):
        ""
        columnWidths = [(0,20),(1,200)]
        for columnId,width in columnWidths:
            self.setColumnWidth(columnId,width)


        

