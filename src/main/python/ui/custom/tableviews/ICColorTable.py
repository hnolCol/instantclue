from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..utils import clearLayout, getStandardFont, BuddyLabel
from ...utils import HOVER_COLOR, createSubMenu, createMenu, createLabel, createTitleLabel, createLineEdit

from ...delegates.quickSelectDelegates import DelegateColor#, ItemDelegate #borrow delegate

import pandas as pd
import numpy as np
import os



class ItemDelegate(QStyledItemDelegate):
    def __init__(self,parent):
        super(ItemDelegate,self).__init__(parent)
    
    def paint(self, painter, option, index):
        painter.setFont(getStandardFont())
        rect = option.rect
        if self.parent().mouseOverItem is not None and index.row() == self.parent().mouseOverItem:
            b = QBrush(QColor(HOVER_COLOR))
            painter.setBrush(b)
            painter.setPen(Qt.NoPen)
            painter.drawRect(option.rect)
            self.addText(index,painter,rect)
        
        else:
            self.addText(index,painter,rect)
    
    def addText(self,index,painter,rect):
        ""
        painter.setPen(QPen(self.parent().model().data(index,Qt.ForegroundRole)))
        rect.adjust(9,0,0,0)
        painter.drawText(rect,   Qt.AlignVCenter | Qt.AlignLeft, self.parent().model().data(index,Qt.DisplayRole))
       
    def setEditorData(self,editor,index):
        editor.setFont(getStandardFont())
        editor.setAutoFillBackground(True)
        editor.setText(self.parent().model().data(index,Qt.DisplayRole))

class ICColorSizeTableBase(QWidget):
    selectionChanged = pyqtSignal()

    def __init__(self, mainController, *args,**kwargs):

        super(ICColorSizeTableBase,self).__init__(*args,**kwargs)
        self.mC = mainController
        self.title = ""

        self.setMaximumHeight(0)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)

    def setTitle(self,title):
        ""
        if isinstance(title,str):
            self.titleLabel.setText(title)
            self.title = title
    
    def setData(self,data, title=None, isEditable = True):
        "" 
        if isinstance(data,pd.DataFrame):
            self.setTitle(title)
            self.table.model().layoutAboutToBeChanged.emit()
            self.table.model().isEditable = isEditable
            self.table.createMenu()
            self.table.model().initData(data)
            self.table.model().completeDataChanged()
            self.table.model().layoutChanged.emit()
            self.setWidgetHeight()
            
    def setWidgetHeight(self):
        ""
        rowCount = self.table.model().rowCount()
        if rowCount == 0:
            maxHeight = 0
        else:
            maxHeight = int(95 + rowCount * self.table.rowHeight)
     
        self.setMaximumHeight(maxHeight)
    
    def reset(self):
        ""
        self.table.model().layoutAboutToBeChanged.emit()
        self.table.model().resetView()
        self.table.model().layoutChanged.emit()
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

class ICColorTable(ICColorSizeTableBase):
    clorMapChanged = pyqtSignal() 
    def __init__(self, *args,**kwargs):

        super(ICColorTable,self).__init__(*args,**kwargs)
        
        self.selectionChanged.connect(self.updateColorInGraph)
        self.clorMapChanged.connect(self.updateColorsByColorMap)
        self.__controls()
        self.__layout()
        

    def __controls(self):
        ""
        self.mainHeader = createTitleLabel("Colors",fontSize = 14)
        self.mainHeader.setWordWrap(True)

        self.titleEdit = createLineEdit("Enter title ..","Title for the legend. Source data remain uneffected.")
        self.titleEdit.hide() # Hide line edit
        self.titleEdit.editingFinished.connect(self.titleChanged)
        self.titleLabel = BuddyLabel(self.titleEdit) # Create our custom label, and assign myEdit as its buddy
        self.titleLabel.setText(self.title)
        self.titleLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed) # Change vertical size policy so they both match and you don't get popping when switching


        #self.titleLabel = createLabel(text = self.title)
        self.table = ColorTable(parent = self, mainController=self.mC)
        self.model = ColorTableModel(parent=self.table)
        self.table.setModel(self.model)

        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch) 
        self.table.resizeColumns()
        self.table.setItemDelegateForColumn(0,DelegateColor(self.table))
        self.table.setItemDelegateForColumn(1,ItemDelegate(self.table))

    def __layout(self):
        ""
        hLabelLayout = QHBoxLayout()
        hLabelLayout.addWidget(self.titleLabel)
        hLabelLayout.addWidget(self.titleEdit)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mainHeader)
        self.layout().addLayout(hLabelLayout)
        self.layout().addWidget(self.table)  

    def titleChanged(self):
        ""
        if not self.titleEdit.text():
            self.titleEdit.hide()
            self.titleLabel.setText(self.title)
            self.titleLabel.show()
        else:
            self.title = self.titleEdit.text()
            self.titleEdit.hide()
            self.titleLabel.setText(self.titleEdit.text())
            self.titleLabel.show()

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
    
    def updateColorInGraph(self):
        ""
        exists, graph =  self.mC.getGraph()
        try:
            if exists:
                graph.setHoverObjectsInvisible()
                graph.updateGroupColors(self.table.model().getLabels(),self.table.model().getItemChangedInternalID())
        except Exception as e:
            print(e) 

    def addLegendToGraph(self, ignoreNaN = False, legendKwargs = {}):
        "Adds color legend to graph."
        exists, graph =  self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.addColorLegendToGraph(
                            self.model._labels.loc[self.model._inLegend.values],
                            ignoreNaN = ignoreNaN, 
                            title = self.title, 
                            legendKwargs = legendKwargs)
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
       

class ColorTableModel(QAbstractTableModel):
    
    def __init__(self, labels = pd.DataFrame(), parent=None, isEditable = False):
        super(ColorTableModel, self).__init__(parent)
        self.initData(labels)
        self.isEditable = isEditable

    def initData(self,labels):

        self._labels = labels
        self._inputLabels = labels.copy()
        self._inLegend = pd.Series(np.ones(shape = self._labels.index.size).astype(bool) ,index=self._labels.index)

    def rowCount(self, parent=QModelIndex()):
        
        return self._labels.index.size

    def columnCount(self, parent=QModelIndex()):
        
        return 2
    
    def dataAvailable(self):
        ""
        return self._labels.index.size > 0 

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

    def getColor(self, tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        return self._labels.loc[dataIndex,"color"]

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

    def toggleInLegend(self,*args,**kwargs):
        ""
        mouseOverItem = self.parent().mouseOverItem
        if mouseOverItem is not None:
            self._inLegend.iloc[mouseOverItem] = not self._inLegend.iloc[mouseOverItem] 

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
            if index.column() != 1:
                return False
            newValue = str(value)
            oldValue = str(self._labels.iloc[index.row()])
            if oldValue != newValue:
                self.updateGroupData(newValue,index)
                self.dataChanged.emit(index,index)
               
            return True

    def setColor(self, dataIndex, hexColor):
        ""
        
        self._labels.loc[dataIndex,"color"] = hexColor
        

    def data(self, index, role=Qt.DisplayRole): 
        ""
        
        if not index.isValid(): 

            return QVariant()
            
        elif role == Qt.DisplayRole and index.column() == 1: 

            return str(self._labels.iloc[index.row(),index.column()])
        
        elif role == Qt.FontRole:

            return getStandardFont()

        elif role == Qt.ForegroundRole and index.column() == 1:
            
            if self._inLegend.iloc[index.row()]:

                return QColor("black")
            else:
                return QColor("grey")

        elif role == Qt.ToolTipRole:

            if index.column() == 0:
                return "Set color. Left-click will cycle through the nan Color (settings) and default color.\nNot available for numeric scales."
            elif index.column() == 1:
                return "Color encoded categorical or numerical values. Double click + cmd/ctrl+c to copy entry."

        elif self.parent().mouseOverItem is not None and role == Qt.BackgroundRole and index.row() == self.parent().mouseOverItem:
            return QColor(HOVER_COLOR)
            
    def flags(self, index):
        "Set Flags of Column"
       
        if self.isEditable and index.column() == 1:
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



class ColorTable(QTableView):

    def __init__(self, parent=None, rowHeight = 22, mainController = None):

        super(ColorTable, self).__init__(parent)
       
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
        self.colorChangedForItem = None
        
        p = self.palette()
        p.setColor(QPalette.Highlight,QColor(HOVER_COLOR))
        p.setColor(QPalette.HighlightedText, QColor("black"))
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
        
        
        
        if self.model() is not None and hasattr(self.model(),"isEditable") and self.model().isEditable: #if editable - add the option to choose color from palette
            menu = createSubMenu(None,["Subset by ..","Color from palette","Add Legend at ..","Add Legend at (-NaN Color) .."])
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
        else:
            menu = createSubMenu(None,["Add Legend at ..","Add Legend at (-NaN Color) .."])

        for legendLoc in legendLocations:
            menu["Add Legend at .."].addAction(legendLoc,lambda lloc = legendLoc: self.parent().addLegendToGraph(legendKwargs = {"loc":lloc}))
            menu["Add Legend at (-NaN Color) .."].addAction(legendLoc,lambda lloc = legendLoc: self.parent().addLegendToGraph(ignoreNaN = True,legendKwargs = {"loc":lloc}))
        
        menu["main"].addAction("Hide/Show in Legend", self.model().toggleInLegend)
        menu["main"].addAction("Remove", self.parent().removeFromGraph)
        menu["main"].addAction("Copy to clipboard",self.parent().copyToClipboard)
        menu["main"].addAction("Save to xlsx",self.parent().saveModelDataToExcel)
        if "Subset by .." in menu:
            menu["Subset by .."].addAction("Group", self.parent().subsetSelection)
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
                    color = QColorDialog(parent=self.parent()).getColor()
                    if color.isValid():
                        self.model().setColor(dataIndex,color.name())
                    else:
                        return
                self.colorChangedForItem = tableIndex.row()
                #emit change signal
                self.parent().selectionChanged.emit()
                self.model().rowDataChanged(tableIndex.row())

                self.colorChangedForItem = None
                
                
            elif tableIndexCol == 1 and self.rightClick:
                self.rightClickedRowIndex = tableIndex.row()
                self.menu.exec(QCursor.pos() + QPoint(4,4))
                
                self.clickedRow = None 
            
            elif tableIndexCol == 1:
                #allow editing
                super().mouseReleaseEvent(e)   

        except Exception as e:
            print(e)


    def mouseMoveEvent(self,event):
        
        ""
        if self.state() == QAbstractItemView.EditingState:
            return 
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


        

