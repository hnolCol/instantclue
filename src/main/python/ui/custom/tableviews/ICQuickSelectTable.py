from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ..utils import getStandardFont
from ...utils import getHoverColor, createSubMenu, createLabel, createTitleLabel

from ...delegates.ICQuickSelect import DelegateColor
from ...delegates.ICSpinbox import SpinBoxDelegate #borrow delegate
from .ICTableBase import ICTableBase, ICModelBase

import pandas as pd
import numpy as np
import os

 

class ICQuickSelectTable(ICTableBase):
    clorMapChanged = pyqtSignal() 
    #selectionChanged signal defined in ICTableBase
    def __init__(self, *args,**kwargs):

        super(ICQuickSelectTable,self).__init__(*args,**kwargs)
        
        self.selectionChanged.connect(self.updateColorAndSizeInGraph)
        self.clorMapChanged.connect(self.updateColorsByColorMap)
        self.__controls()
        self.__layout()
        

    def __controls(self):
        ""
        self.mainHeader = createTitleLabel("QuickSelect",fontSize = 14)
        self.mainHeader.setWordWrap(True)
        self.titleLabel = createLabel(text = self.title)
        self.table = QuickSelectTable(parent = self, mainController=self.mC)
        self.model = QuickSelectTableModel(parent=self.table)
        self.table.setModel(self.model)

        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Fixed) 
        self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch) 
        self.table.resizeColumns()
        self.table.setItemDelegateForColumn(0,DelegateColor(self.table))
        self.table.setItemDelegateForColumn(1,SpinBoxDelegate(self.table))

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mainHeader)
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.table)  

    @pyqtSlot()
    def updateColorsByColorMap(self):
        ""
        
        if self.model.rowCount() > 0:
            self.table.createMenu()
            if self.mC.getPlotType() == "scatter":
                funcProps = {"key":"plotter:getScatterColorGroups","kwargs":{"dataID":self.mC.getDataID(),
                    "colorColumn":None,
                    "colorColumnType":None,
                    "colorGroupData":self.model._labels}}
            
                self.mC.sendRequestToThread(funcProps)
            else:
                colorList = self.mC.colorManager.getNColorsByCurrentColorMap(N = self.model.rowCount())
                self.model.updateColors(colorList)
    
    @pyqtSlot()
    def updateColorAndSizeInGraph(self):
        ""
        exists, graph =  self.mC.getGraph()
        try:
            if exists:
                graph.setHoverObjectsInvisible()
                graph.updateQuickSelectData(self.table.model().getLabels(),self.table.model().getItemChangedInternalID())
        except Exception as e:
            print(e) 

    def addLegendToGraph(self, ignoreNaN = False, legendKwargs = {}):
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.addQuickSelectLegendToGraph(self.model._labels,ignoreNaN = ignoreNaN, title = self.title, legendKwargs = legendKwargs)

    def removeFromGraph(self):
        ""
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.setLegendInvisible()
            graph.setQuickSelectScatterInvisible()
            #update figure and reset table
            graph.updateFigure.emit()

        
        self.reset()
    

class QuickSelectTableModel(ICModelBase):
    
    def __init__(self, labels = pd.DataFrame(), parent=None, isEditable = False):
        super(QuickSelectTableModel, self).__init__(parent)
        self.initData(labels)
        self.isEditable = isEditable

    def initData(self,labels):
        
        self._labels = labels
        self.maxSize = None
        self._inputLabels = labels.copy()
        self.setDefaultSize()
        self.setRowHeights()

    def columnCount(self, parent=QModelIndex()):
        "Three Columns to show: color, size, label"
        return 3
    
    def setDefaultSize(self,size=50):
        ""
        self.defaultSize = size

    def updateData(self,value,index):
        ""
        dataIndex = self.getDataIndex(index.row())
        if dataIndex is not None:
            self._labels[dataIndex] = value
            self._inputLabels = self._labels.copy()

    def deleteEntriesByIndexList(self,indexList):
        ""
        dataIndices = [self.getDataIndex(tableIndex.row()) for tableIndex in indexList]
        self.layoutAboutToBeChanged.emit()
        self._labels = self._labels.drop(dataIndices)
        self._inputLabels = self._labels.copy()
        self.layoutChanged.emit()
        self.completeDataChanged()

    def deleteEntry(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex in self._inputLabels.index:
            self.layoutAboutToBeChanged.emit()
            self._labels = self._labels.drop(dataIndex)
            self._inputLabels = self._labels
            self.layoutChanged.emit()
            self.completeDataChanged()

    def getColor(self, tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        return self._labels.loc[dataIndex,"color"]

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
            if index.column() > 1:
                return False
            elif index.column() == 0:
         
                return False

            elif index.column() == 1:
                
                #get value
                v = int(np.sqrt(value))
                self._labels.iloc[index.row(),index.column()] = value
                
                if v > self.maxSize:
                    self.setRowHeights()
                elif v < self.minSize:
                    self.setRowHeights()
                self.parent().sizeChangedForItem = index.row()
                self.parent().parent().selectionChanged.emit()

                if self._labels.index.size == 1:
                    rowHeight = v + self.parent().rowHeight
                    self.parent().setRowHeight(index.row(),rowHeight)
                    self.parent().setColumnWidth(0,rowHeight)
                
                self.completeDataChanged()
                return True

    def setRowHeights(self):
        ""
        if "size" in self._labels.columns and self.rowCount() > 0:
            
            vs = np.sqrt(self._labels["size"].values)
            self.maxSize = np.nanmax(vs)
            self.minSize = np.nanmin(vs)
        else:
            self.minSize = 10
            self.maxSize = 200

    def setColor(self, dataIndex, hexColor):
        ""
        self._labels.loc[dataIndex,"color"] = hexColor
        
    def setColorForAllIdcs(self,hexColor):
        ""
        self._labels.loc[:,"color"] = hexColor 


    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
        
        if not index.isValid(): 

            return QVariant()
            
        elif role == Qt.ItemDataRole.DisplayRole and index.column() == 2: 
            return str(self._labels.iloc[index.row(),2])
        
        elif role == Qt.ItemDataRole.FontRole:

            return getStandardFont()

        elif role == Qt.ItemDataRole.ToolTipRole:

            if index.column() == 0:
                return """Color of quick select item.\n
                         Changing the color here (richt-click), will adjust the graph item color,\n
                         but will not change the color in the QuickSelect widget."""
            elif index.column() == 1:
                return """Color of quick select item.\n
                         Changing the size here (double-click), will adjust the graph item size,\n
                         but will not change the color in the QuickSelect widget."""

        elif role == Qt.ItemDataRole.EditRole and index.column() == 1:

            return self._labels.iloc[index.row(),index.column()]

        elif self.parent().mouseOverItem is not None and role == Qt.ItemDataRole.BackgroundRole and index.row() == self.parent().mouseOverItem:
            return QColor(getHoverColor())
            
    def flags(self, index):
        "Set Flags of Column"
        if index.column() < 2 and self.isEditable:
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable
        else:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


    def search(self,searchString):
        ""
        if self._inputLabels.size == 0:
            return
        self.layoutAboutToBeChanged.emit()
        if len(searchString) > 0:
            boolMask = self._labels.str.contains(searchString,case=False,regex=False)
            self._labels = self._labels.loc[boolMask]
        else:
            self._labels = self._inputLabels
        self.layoutChanged.emit()
        self.completeDataChanged()

    def getInitColor(self,dataIndex):
        ""
        return self._inputLabels.loc[dataIndex,"color"]

    def setInitColor(self,dataIndex):
        ""
        initColor = self._inputLabels.loc[dataIndex,"color"]
        self.setColor(dataIndex,initColor)


    def updateColors(self,colorList):
        ""
        
        if len(colorList) == self._labels.index.size:

            nanObject = self.parent().mC.config.getParam("replaceObjectNan")
            nanColor = self.parent().mC.config.getParam("nanColor")
            idxWithNaNColor = self._labels.index[self._labels["color"] == nanColor]
            self._labels["color"] = colorList
            self._labels.loc[idxWithNaNColor,"color"] = nanColor
            self._inputLabels["color"] = colorList
            for nanString in ["NaN",nanObject]:
                if nanString in self._labels["group"].values:
                    idx = self._labels.index[self._labels["group"] == nanString]
                    self._labels.loc[idx,"color"] = nanColor 
            
            self.completeDataChanged()
            #emi signal to widget to update colors in graph
            self.parent().parent().selectionChanged.emit()



class QuickSelectTable(QTableView):

    def __init__(self, parent=None, rowHeight = 22, mainController = None):

        super(QuickSelectTable, self).__init__(parent)
       
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
        self.sizeChangedForItem = None
        
        p = self.palette()
        p.setColor(QPalette.ColorRole.Highlight,QColor(getHoverColor()))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor("black"))
        self.setPalette(p)

        self.setStyleSheet("""QTableView {border:None};""")

    def colorChangedFromMenu(self,event=None, hexColor = ""):
        ""
        rowIndex = self.rightClickedRowIndex
        
        dataIndex = self.model().getDataIndex(rowIndex)
        self.model().setColor(dataIndex,hexColor)
        self.parent().selectionChanged.emit()
        self.model().rowDataChanged(rowIndex)

    def selectSingleColor(self):
        "Select a single color for all items in the table"
        color = QColorDialog(parent=self.parent()).getColor()
        if color.isValid():
            self.model().setColorForAllIdcs(color.name())
            self.parent().selectionChanged.emit()

    def createMenu(self):
        ""
        legendLocations = ["upper right","upper left","center left","center right","lower left","lower right"]
        menu = createSubMenu(None,["Subset by ..","Color from palette","Add Legend at ..","Add Legend at (-NaN Color) .."])
        menu["main"].addAction("Single color for all items", self.selectSingleColor)
        
        colors = self.mC.colorManager.getNColorsByCurrentColorMap(8)
        for col in colors:
            pixmap = QPixmap(20,20)
            pq = QPainter(pixmap) 
            pq.setBrush(QColor(col))
            pq.drawRect(0,0,20,20)
            action = menu["Color from palette"].addAction(col, lambda col = col: self.colorChangedFromMenu(hexColor = col))
            i = QIcon() 
            i.addPixmap(pixmap)
            pq.end()
            action.setIcon(i)
        
        for legendLoc in legendLocations:
            menu["Add Legend at .."].addAction(legendLoc,lambda lloc = legendLoc: self.parent().addLegendToGraph(legendKwargs = {"loc":lloc}))
            menu["Add Legend at (-NaN Color) .."].addAction(legendLoc,lambda lloc = legendLoc: self.parent().addLegendToGraph(ignoreNaN = True,legendKwargs = {"loc":lloc}))
        
        menu["main"].addAction("Save to xlsx",self.parent().saveModelDataToExcel)
        menu["Subset by .."].addAction("Group", self.parent().subsetSelection)
        menu["main"].addAction("Remove", self.parent().removeFromGraph)
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
        if e.buttons() == Qt.MouseButton.RightButton:
            self.rightClick = True
        else:
            self.rightClick = False
            tableIndex = self.mouseEventToIndex(e)
            if tableIndex.column() == 1: #forward press event
                super().mousePressEvent(e)
    
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
                if not self.rightClick:
                    nanColor = self.mC.config.getParam("nanColor")
                    currentColor = self.model().getColor(tableIndex)
                    
                    if currentColor == nanColor: 
                        #if current color is nan color , set init color, if init color is also nan, just dont do anything
                        if self.model().getInitColor(dataIndex) == nanColor:
                            return
                        self.model().setInitColor(dataIndex)
                    else:
                        self.model().setColor(dataIndex,nanColor)
                else:
                    color = QColorDialog.getColor()
                    if color.isValid():
                        self.model().setColor(dataIndex,color.name())
                    else:
                        return
                self.colorChangedForItem = tableIndex.row()
                #emit change signal
                self.parent().selectionChanged.emit()
                self.model().rowDataChanged(tableIndex.row())

                self.colorChangedForItem = None
            
                
            elif tableIndexCol == 2 and self.rightClick:
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
        self.model().completeDataChanged()
 
    def resizeColumns(self):
        ""
        columnWidths = [(0,20),(1,40),(2,50)]
        for columnId,width in columnWidths:
            self.setColumnWidth(columnId,width)


        

