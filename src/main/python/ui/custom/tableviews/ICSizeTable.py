from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import clearLayout, getStandardFont
from ...utils import HOVER_COLOR, createSubMenu, createMenu, createLabel, createTitleLabel
from ...delegates.spinboxDelegate import SpinBoxDelegate #borrow delegate
from .ICColorTable import ICColorSizeTableBase
import pandas as pd
import numpy as np

class ICSizeTable(ICColorSizeTableBase):
    
    def __init__(self,*args,**kwargs):

        super(ICSizeTable,self).__init__(*args,**kwargs)
        self.selectionChanged.connect(self.updateSizeInGraph)
        
        self.__controls()
        self.__layout()

    def __controls(self):
        ""
        self.mainHeader = createTitleLabel("Sizes",fontSize = 14)
        self.titleLabel = createLabel(text = self.title)
        self.table = SizeTable(parent = self, mainController=self.mC)
        df = pd.DataFrame(np.array([[1,3],[2,5]]))
        self.model = SizeTableModel(parent=self.table,labels=df)
        self.table.setModel(self.model)

        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch) 
        self.table.resizeColumns()
        self.table.setItemDelegateForColumn(0,SpinBoxDelegate(self.table))
        
        
    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mainHeader)
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.table)

    def resetSizeInGraph(self):
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.resetSize()
        

    def updateSizeInGraph(self):
        ""
        exists, graph =  self.mC.getGraph()
        try:
            if exists:
                graph.setHoverObjectsInvisible()
                graph.updateGroupSizes(self.table.model().getLabels(),self.table.model().getItemChangedInternalID())
        except Exception as e:
            print(e)
        
    def addLegendToGraph(self, ignoreNaN = False, legendKwargs = {}):
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.addSizeLegendToGraph(self.model._labels,ignoreNaN,title = self.title, legendKwargs = legendKwargs)
            self.model.completeDataChanged()


class SizeTableModel(QAbstractTableModel):
    
    def __init__(self, labels = pd.DataFrame(), parent=None, isEditable = False):
        super(SizeTableModel, self).__init__(parent)
        self.initData(labels)
        self.isEditable = isEditable

    def initData(self,labels):

        self._labels = labels
        self._inputLabels = labels.copy()
        
        self.setDefaultSize()
        self.maxSize = None
        self.setRowHeights()
        

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

    def updateGroupData(self,value,index):
        ""
        dataIndex = self.getDataIndex(index.row())
        if dataIndex is not None:
            self._labels.loc[dataIndex,"group"] = value
            self._inputLabels = self._labels.copy()

    def validDataIndex(self,row):
        ""
        return row <= self._labels.index.size - 1

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

    def getInternalIDByRowIndex(self,row):
        ""
        return self._labels.iloc[row].loc["internalID"]
 
    def getGroupByInternalID(self,internalID):
        ""
        if "internalID" in self._labels.columns:
            boolIdx = self._labels["internalID"] == internalID
            if np.any(boolIdx):
                return self._labels.loc[boolIdx,"group"].values[0]

    def getItemChangedInternalID(self):
        ""
        if self.parent().sizeChangedForItem is not None and "internalID" in self._labels.columns:
            return self._labels.iloc[self.parent().sizeChangedForItem ].loc["internalID"]
            
    def setData(self,index,value,role):
        ""
        
        row =index.row()
        indexBottomRight = self.index(row,self.columnCount())
        if role == Qt.UserRole:
            self.dataChanged.emit(index,indexBottomRight)
            return True
        if role == Qt.CheckStateRole:
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True
        elif role == Qt.EditRole:
            if not self.isEditable:
                return False
            #get value
            if index.column() == 0:
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
            elif index.column() == 1:
                newValue = str(value)
                oldValue = str(self._labels.iloc[index.row()])
                if oldValue != newValue:
                    self.updateGroupData(newValue,index)
                    self.dataChanged.emit(index,index)
            else:
                return False
            
            return True

    def setRowHeights(self):
        ""
        if "size" in self._labels.columns:
            vs = np.sqrt(self._labels["size"].values)
            self.maxSize = np.nanmax(vs)
            self.minSize = np.nanmin(vs)
           

    def data(self, index, role=Qt.DisplayRole): 
        ""
        
        if not index.isValid(): 

            return QVariant()

        elif role == Qt.EditRole:

            return self._labels.iloc[index.row(),index.column()]
            
        elif role == Qt.DisplayRole and index.column() == 1: 
            return str(self._labels.iloc[index.row(),index.column()])
        
        elif role == Qt.FontRole:

            return getStandardFont()

        elif role == Qt.ToolTipRole:

            dataIndex = self.getDataIndex(index.row())
            if index.column() == 0:
                if self.isEditable:
                    return "Current size: {}\nDouble click allows user-defined sizes.".format(self._labels.loc[dataIndex,"size"])
                else:
                    return "Current size: {}\nSize range for numeric values can only be changed in the settings dialog.".format(self._labels.loc[dataIndex,"size"])
            elif index.column() == 1:
                return "Current size: {}\nSize encoded categorical or numeric values.\nDouble click to change the name (categorical columns only). This does not effect the source data but allows to change the legend label.".format(self._labels.loc[dataIndex,"size"])

        elif self.parent().mouseOverItem is not None and role == Qt.BackgroundRole and index.row() == self.parent().mouseOverItem:

            return QColor(HOVER_COLOR)
            
    def flags(self, index):
        "Set Flags of Column"

        if self.isEditable:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsEnabled 

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
        self._labels = pd.Series()
        self._inputLabels = self._labels.copy()
        self.completeDataChanged()




class SizeTable(QTableView):

    def __init__(self, parent=None, rowHeight = 22, mainController = None):

        super(SizeTable, self).__init__(parent)
       
        self.setMouseTracking(True)
        self.setShowGrid(True)
        self.verticalHeader().setDefaultSectionSize(rowHeight)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)

        self.mainController = mainController
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 

        self.createMenu()

        self.rowHeight      =   rowHeight
        self.rightClick     =   False
        self.mouseOverItem  =   None
        self.sizeChangedForItem = None
        
        p = self.palette()
        p.setColor(QPalette.Highlight,QColor(HOVER_COLOR))
        p.setColor(QPalette.HighlightedText, QColor("black"))
        self.setPalette(p)

        self.setStyleSheet("""QTableView {background-color: #F6F6F6;border:None};""")


    def createMenu(self):
        ""
        legendLocations = ["upper right","upper left","center left","center right","lower left","lower right"]
        menu = createSubMenu(None,["Subset by ..","Add Legend at .."])
        menu["main"].addAction("Remove",self.parent().resetSizeInGraph)
        menu["Subset by .."].addAction("Group", self.parent().subsetSelection)
        menu["main"].addAction("Save to xlsx",self.parent().saveModelDataToExcel)

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
            if tableIndexCol == 0:
                #dataIndex = self.model().getDataIndex(tableIndex.row())
                if not self.rightClick:
                    return
                    #stdSize = self.parent().mC.config.getParam("scatterSize")
                else:
                    return

                self.sizeChangedForItem = tableIndex.row()
                self.parent().selectionChanged.emit()
                self.model().rowDataChanged(tableIndex.row())
                
            elif tableIndexCol == 1 and self.rightClick:
                self.rightClickedRowIndex = tableIndex.row()
                self.menu.exec(QCursor.pos())
                
                self.rightClick = False
            else:
                super().mouseReleaseEvent(e)

        except Exception as e:
            print(e)

    def getSize(self):
        ""


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
        columnWidths = [(0,40),(1,200)]
        for columnId,width in columnWidths:
            self.setColumnWidth(columnId,width)


        

