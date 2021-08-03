from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import clearLayout, getStandardFont
from ...utils import HOVER_COLOR, createSubMenu, createMenu, createLabel, createTitleLabel
from ...delegates.spinboxDelegate import SpinBoxDelegate #borrow delegate
from .ICColorTable import ICColorSizeTableBase
from .ICDataTable import ICLabelDataTableDialog
import pandas as pd
import numpy as np

class ICLabelTable(ICColorSizeTableBase):
    
    def __init__(self,header = "Labels", *args,**kwargs):

        super(ICLabelTable,self).__init__(*args,**kwargs)
        self.selectionChanged.connect(self.updateLabelInGraph)
        self.header = header
        self.__controls()
        self.__layout()

    def __controls(self):
        ""
        self.mainHeader = createTitleLabel(self.header,fontSize = 14)
        self.titleLabel = createLabel(text = self.title)
        self.table = LabelTable(parent = self, mainController=self.mC)
        df = pd.DataFrame(np.array([[1,3],[2,5]]))
        self.model = LabelTableModel(parent=self.table,labels=df)
        self.table.setModel(self.model)

        #self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch) 
        self.table.resizeColumns()
        #self.table.setItemDelegateForColumn(0,SpinBoxDelegate(self.table))
        
        
    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mainHeader)
        self.layout().addWidget(self.titleLabel)
        self.layout().addWidget(self.table)

    def hideLabel(self):
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            columnName = self.model.getCurrentGroup()
            graph.removeColumnNameFromTooltip(columnName)

    def removeLabelsFromGraph(self):
        "Removes Annotations from Graph"
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.removeAnnotationsFromGraph()
            
    def updateLabelInGraph(self):
        ""
        exists, graph =  self.mC.getGraph()
        try:
            if exists:
                print("upadintg this blody graph?")
        except Exception as e:
            print(e)

    def setLabelInAllPlots(self):
        "Enables labeling data rows in multiple graphs (e.g. scatter)"
        exists, graph =  self.mC.getGraph()
        try:
            if exists:
                graph.setLabelInAllPlots()
        except Exception as e:
            print(e)
    
    def showAnnotationsInDataTable(self):
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            idxByAx = graph.getAnnotationIndices()
            nAxes = graph.getNumberOfAxes()
            newColumnNames = ["{}:{}_{}".format(ax.get_xlabel(),ax.get_ylabel(),n) for n,ax in enumerate(idxByAx.keys())]
            idxMapper = dict([(newColumnNames[n],idx) for  n,idx in enumerate(idxByAx.values())])
            columnNames = self.model.getLabels().values.flatten().tolist()
            #get data by column Names (label data)
            data = self.mC.data.getDataByColumnNames(self.mC.getDataID(),columnNames)["fnKwargs"]["data"]
            dataIndices = data.index
            #pool the annotation data
            data = data.values.astype(np.str)
            if data.shape[1] > 1:
                data = np.apply_along_axis(' ; '.join, 1, data)
            if nAxes > 1:
                data = np.tile(data,(1,nAxes))
            dlg = ICLabelDataTableDialog(df = pd.DataFrame(data,columns=newColumnNames,index=dataIndices), mainController=self.mC, modelKwargs = {"selectionCallBack":self.annotationSelected}, tableKwargs = {"onHoverCallback":self.onHover})
            dlg.model.setCheckStateByColumnNameAndIndex(idxMapper)
            dlg.exec_() 

    def onHover(self,tableRow,dataIndex):
        ""
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverData([dataIndex])

    def annotationSelected(self,dataIndex,columnIndex):
        ""
        _, graph =  self.mC.getGraph()
        if hasattr(graph,"annotateInAxByDataIndex"):
            graph.annotateInAxByDataIndex(columnIndex,dataIndex)
            graph.updateFigure.emit() 

    def removeFromGraph(self):
        "Remove labels/tooltips from graph"
        exists, graph =  self.mC.getGraph()
        if exists:
            if self.header == "Labels":
                graph.removeLabels()
            elif self.header == "Tooltip":
                graph.removeTooltip()
            self.reset()
    

class LabelTableModel(QAbstractTableModel):
    
    def __init__(self, labels = pd.DataFrame(), parent=None):
        super(LabelTableModel, self).__init__(parent)
        self.initData(labels)

    def initData(self,labels):
        
        self._labels = labels
        self._inputLabels = labels.copy()
        self.setDefaultSize()

    def rowCount(self, parent=QModelIndex()):
        
        return self._labels.index.size

    def columnCount(self, parent=QModelIndex()):
        
        return 1
    
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

    def updateData(self,value,index):
        ""
        dataIndex = self.getDataIndex(index.row())
        if dataIndex is not None:
            self._labels[dataIndex] = value
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
            return self._labels.iloc[mouseOverItem].loc["columnName"]

    def getCurrentInternalID(self):
        ""
        mouseOverItem = self.parent().mouseOverItem
        if mouseOverItem is not None and "internalID" in self._labels.columns: 
            return self._labels.iloc[mouseOverItem].loc["internalID"]


    def getItemChangedInternalID(self):
        ""
        if self.parent().sizeChangedForItem is not None and "internalID" in self._labels.columns:
            return self._labels.iloc[self.parent().sizeChangedForItem ].loc["internalID"]
            
    def setData(self,index,value,role):
        ""
        row = index.row()

        indexBottomRight = self.index(row,self.columnCount())
        if role == Qt.UserRole:
            self.dataChanged.emit(index,indexBottomRight)
            return True
        if role == Qt.CheckStateRole:
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True

    def data(self, index, role=Qt.DisplayRole): 
        ""
       
        if not index.isValid(): 

            return QVariant()

        elif role == Qt.EditRole:
            return self._labels.iloc[index.row(),index.column()]
            
        elif role == Qt.DisplayRole and index.column() == 0: 
            return str(self._labels.iloc[index.row(),index.column()])
        
        elif role == Qt.FontRole:

            return getStandardFont()

        elif role == Qt.ToolTipRole:
            dataIndex = self.getDataIndex(index.row())
            if index.column() == 0:
                return str(self._labels.iloc[index.row(),index.column()])
            elif index.column() == 1:
                return "kk"

        elif self.parent().mouseOverItem is not None and role == Qt.BackgroundRole and index.row() == self.parent().mouseOverItem:
            return QColor(HOVER_COLOR)
        
    def flags(self, index):
        "Set Flags of Column"
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled 
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

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



class LabelTable(QTableView):

    def __init__(self, parent=None, rowHeight = 22, mainController = None):

        super(LabelTable, self).__init__(parent)
       
        self.setMouseTracking(True)
        self.setShowGrid(True)
        self.verticalHeader().setDefaultSectionSize(rowHeight)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)

        self.mC = mainController
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 

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
        menu = createSubMenu(None,[])
        parent = self.parent() 
        menu["main"].addAction("Disable",parent.removeFromGraph)
        
        if parent.header == "Labels":
            _, graph = self.mC.getGraph()
            if hasattr(graph,"isAnnotationInAllPlotsEnabled"):
                enabled = graph.isAnnotationInAllPlotsEnabled()
                menu["main"].addAction("{} annotations in all subplots".format("Disable" if enabled else "Enable"),self.parent().setLabelInAllPlots)
                menu["main"].addAction("Remove Labels",parent.removeLabelsFromGraph)
            menu["main"].addAction("Select Annotations in Data",parent.showAnnotationsInDataTable)
            menu["main"].addAction("Save to xlsx",parent.saveModelDataToExcel)
        else:
            menu["main"].addAction("Hide selected label",parent.hideLabel)
            

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
                
                
            if tableIndexCol == 0 and self.rightClick:
                #idx = self.model().index(0,0)
                pos = QCursor.pos()
                pos += QPoint(3,3)
                self.createMenu()
                self.menu.exec(pos)
                
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
        self.model().completeDataChanged()
 
    def resizeColumns(self):
        ""
        columnWidths = [(0,40),(1,200)]
        for columnId,width in columnWidths:
            self.setColumnWidth(columnId,width)


        

