from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ..utils import clearLayout, getStandardFont
from ...utils import getHoverColor, createSubMenu, createMenu, createLabel, createTitleLabel
from ...delegates.ICSpinbox import SpinBoxDelegate #borrow delegate
from .ICTableBase import ICTableBase, ICModelBase

from ...dialogs.ICDataTable import ICLabelDataTableDialog
import pandas as pd
import numpy as np

class ICLabelTable(ICTableBase):
    
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

        #self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch) 
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
        if exists and hasattr(graph,"removeColumnNameFromTooltip"):
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
        if exists and hasattr(graph,"getAnnotationIndices"):
            idxByAx = graph.getAnnotationIndices()
            nAxes = graph.getNumberOfAxes()
            newColumnNames = ["{}:{}_{}".format(ax.get_xlabel(),ax.get_ylabel(),n) for n,ax in enumerate(idxByAx.keys())]
            idxMapper = dict([(newColumnNames[n],idx) for  n,idx in enumerate(idxByAx.values())])
            columnNames = self.model.getLabels().values.flatten().tolist()
            #get data by column Names (label data)
            data = self.mC.data.getDataByColumnNames(self.mC.getDataID(),columnNames)["fnKwargs"]["data"]
            dataIndices = data.index
            #pool the annotation data
            data = data.values.astype(str)
            if data.shape[1] > 1:
                data = np.apply_along_axis(' ; '.join, 1, data)
            if nAxes > 1:
                data = np.tile(data,(1,nAxes))
            dlg = ICLabelDataTableDialog(df = pd.DataFrame(data,columns=newColumnNames,index=dataIndices), mainController=self.mC, modelKwargs = {"selectionCallBack":self.annotationSelected}, tableKwargs = {"onHoverCallback":self.onHover})
            dlg.model.setCheckStateByColumnNameAndIndex(idxMapper)
            dlg.exec() 

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
    

class LabelTableModel(ICModelBase):
    
    def __init__(self, labels = pd.DataFrame(), parent=None):
        super(LabelTableModel, self).__init__(parent)
        self.initData(labels)

    def initData(self,labels):
        
        self._labels = labels
        self._inputLabels = labels.copy()
        self.setDefaultSize()

    def columnCount(self, parent=QModelIndex()):
        
        return 1

    def setDefaultSize(self,size=50):
        ""
        self.defaultSize = size

    def updateData(self,value,index):
        ""
        dataIndex = self.getDataIndex(index.row())
        if dataIndex is not None:
            self._labels[dataIndex] = value
            self._inputLabels = self._labels.copy()

    def getItemChangedInternalID(self):
        ""
        if self.parent().sizeChangedForItem is not None and "internalID" in self._labels.columns:
            return self._labels.iloc[self.parent().sizeChangedForItem ].loc["internalID"]
            
    def setData(self,index,value,role):
        ""
        row = index.row()

        indexBottomRight = self.index(row,self.columnCount())
        if role == Qt.ItemDataRole.UserRole:
            self.dataChanged.emit(index,indexBottomRight)
            return True
        if role == Qt.CheckStateRole:
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True

    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
       
        if not index.isValid(): 

            return QVariant()

        elif role == Qt.ItemDataRole.EditRole:
            return self._labels.iloc[index.row(),index.column()]
            
        elif role == Qt.ItemDataRole.DisplayRole and index.column() == 0: 
            return str(self._labels.iloc[index.row(),index.column()])
        
        elif role == Qt.ItemDataRole.FontRole:

            return getStandardFont()

        elif role == Qt.ItemDataRole.ToolTipRole:
            dataIndex = self.getDataIndex(index.row())
            if index.column() == 0:
                return str(self._labels.iloc[index.row(),index.column()])
            elif index.column() == 1:
                return "kk"

        elif self.parent().mouseOverItem is not None and role == Qt.ItemDataRole.BackgroundRole and index.row() == self.parent().mouseOverItem:
            return QColor(getHoverColor())
        
    def flags(self, index):
        "Set Flags of Column"
        if index.column() == 0:
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled 
        else:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class LabelTable(QTableView):

    def __init__(self, parent=None, rowHeight = 22, mainController = None):

        super(LabelTable, self).__init__(parent)
       
        self.setMouseTracking(True)
        self.setShowGrid(True)
        self.verticalHeader().setDefaultSectionSize(rowHeight)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)

        self.mC = mainController
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) 

        self.rowHeight      =   rowHeight
        self.rightClick     =   False
        self.mouseOverItem  =   None
        self.sizeChangedForItem = None
        
        p = self.palette()
        p.setColor(QPalette.ColorRole.Highlight,QColor(getHoverColor()))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor("black"))
        self.setPalette(p)

        self.setStyleSheet("""QTableView {border:None};""")


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
        if e.buttons() == Qt.MouseButton.RightButton:
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


        

