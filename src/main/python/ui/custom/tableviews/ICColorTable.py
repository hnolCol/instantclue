from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from ..utils import clearLayout, getStandardFont, BuddyLabel
from ...utils import getHoverColor, createSubMenu, createMenu, createLabel, createTitleLabel, createLineEdit, getStdTextColor
from .ICTableBase import ICTableBase, ItemDelegate, ICModelBase
from ...delegates.ICQuickSelect import DelegateColor#, ItemDelegate #borrow delegate
from ...dialogs.ICDataInputDialog import ICDataInput
import pandas as pd
import numpy as np
import os




class ICColorTable(ICTableBase):
    clorMapChanged = pyqtSignal() 
    def __init__(self, *args,**kwargs):

        super(ICColorTable,self).__init__(*args,**kwargs)
        self.colorValueLimit = None
        self.selectionChanged.connect(self.updateColorInGraph)
        self.clorMapChanged.connect(self.updateColorsByColorMap)
        self.currentHighLight = None
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
        self.titleLabel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed) # Change vertical size policy so they both match and you don't get popping when switching
        self.titleLabel.setWordWrap(True)

        #self.titleLabel = createLabel(text = self.title)
        self.table = ColorTable(parent = self, mainController=self.mC)
        self.model = ColorTableModel(parent=self.table)
        self.table.setModel(self.model)

        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch) 
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
                funcProps = {"key":"plotter:getScatterColorGroups",
                            "kwargs":{"dataID":self.mC.getDataID(),
                                    "colorColumn":None,
                                    "colorColumnType":None,
                                    "colorGroupData":self.model._labels,
                                    "userMinMax":self.colorValueLimit}
                                    }
            
                self.mC.sendRequestToThread(funcProps)
            else:
                colorList = self.mC.colorManager.getNColorsByCurrentColorMap(N = self.model.rowCount())
                self.model.updateColors(colorList)
    
    def highlightColorGroup(self, reset = False):
        ""
        exists, graph =  self.mC.getGraph()
        if exists and hasattr(graph,"highlightGroupByColor"):
            colorGroupData = self.table.model().getLabels()
            if reset:
                graph.setHoverObjectsInvisible()
                graph.highlightGroupByColor(colorGroupData,None)
                self.currentHighLight = None 
            else:
                internalID = self.table.model().getCurrentInternalID()
                if internalID != self.currentHighLight:
                    graph.setHoverObjectsInvisible()
                    graph.highlightGroupByColor(colorGroupData,internalID)
                    self.currentHighLight = internalID
               
    @pyqtSlot()
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
    
    def setMinMaxByUser(self,*args,**kwargs):
        ""
        askLimits = ICDataInput(mainController=self.mC, title = "Color map limits for scatter.",valueNames = ["min","max"], valueTypes = {"min":float,"max":float})
        if askLimits.exec():
            minValue, maxValue = askLimits.providedValues["min"], askLimits.providedValues["max"]
            if minValue > maxValue:
                setattr(self,"colorValueLimit",(maxValue,minValue))
            else:
                setattr(self,"colorValueLimit",(minValue,maxValue))
        self.clorMapChanged.emit()

    def setMinMaxByDefault(self,*args,**kwargs):
        ""
        if self.colorValueLimit is not None:
            setattr(self,"colorValueLimit",None)
            self.clorMapChanged.emit()

    def removeFromGraph(self):
        ""
        ""
        exists, graph =  self.mC.getGraph()

        if exists and (graph.hasScatters() or graph.isHclust()):
            graph.setHoverObjectsInvisible()
            graph.setLegendInvisible()
            graph.setNaNColor()
            self.model.completeDataChanged()
        elif exists:
            self.table.setNaNColorForAllItems()
    
    

class ColorTableModel(ICModelBase):
    
    def __init__(self, labels = pd.DataFrame(), parent=None, isEditable = False):
        super(ColorTableModel, self).__init__(parent)
        self.initData(labels)
        self.isEditable = isEditable

    def initData(self,labels):
        "Init data"
        self._labels = labels
        self._inputLabels = labels.copy()
        self._inLegend = pd.Series(np.ones(shape = self._labels.index.size).astype(bool) ,index=self._labels.index)

    def columnCount(self, parent=QModelIndex()):
        
        return 2

    def updateGroupData(self,value,index):
        ""
        dataIndex = self.getDataIndex(index.row())
        if dataIndex is not None:
            self._labels.loc[dataIndex,"group"] = value
            self._inputLabels = self._labels.copy()

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
        if role == Qt.ItemDataRole.UserRole:
            self.dataChanged.emit(index,indexBottomRight)
            return True
        if role == Qt.ItemDataRole.CheckStateRole:
            self.setCheckState(index)
            self.dataChanged.emit(index,indexBottomRight)
            return True
        elif role == Qt.ItemDataRole.EditRole:
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
        
    def setColorForAllIdcs(self,hexColor):
        ""
        self._labels.loc[:,"color"] = hexColor

    def setColorSelectedIdcs(self,hexColor,indexList):
        ""
        self._labels.loc[self.getSelectedData(indexList).index,"color"] = hexColor

    def data(self, index, role=Qt.ItemDataRole.DisplayRole): 
        ""
        
        if not index.isValid(): 

            return QVariant()
            
        elif role == Qt.ItemDataRole.DisplayRole and index.column() == 1: 

            return str(self._labels.iloc[index.row(),index.column()])
        
        elif role == Qt.ItemDataRole.FontRole:

            return getStandardFont()

        elif role == Qt.ItemDataRole.ForegroundRole and index.column() == 1:
            
            if self._inLegend.iloc[index.row()]:

                return QColor("black")
            else:
                return QColor("grey")

        elif role == Qt.ItemDataRole.ToolTipRole:

            if index.column() == 0:
                return "Set color. Left-click will cycle through the nan Color (settings) and default color.\nNot available for numeric scales."
            elif index.column() == 1:
                return "Color encoded categorical or numerical values. Double click + cmd/ctrl+c to copy entry."

        elif self.parent().mouseOverItem is not None and role == Qt.ItemDataRole.BackgroundRole and index.row() == self.parent().mouseOverItem:
            return QColor(getHoverColor())
            
    def flags(self, index):
        "Set Flags of Column"
       
        if self.isEditable and index.column() == 1:
            return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable
        else:
            return Qt.ItemFlag.ItemIsEnabled 
        
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
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) 
        
        

        self.rowHeight      =   rowHeight
        self.rightClick     =   False
        self.mouseOverItem  =   None
        self.colorChangedForItem = None
        
        p = self.palette()
        p.setColor(QPalette.ColorRole.Highlight,QColor(getHoverColor()))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor("black"))
        self.setPalette(p)

        self.setStyleSheet("""QTableView {border:None};""")


    def setNaNColorForAllItems(self):
        "Just set all idcs to items."
        hexNanColor = self.mC.config.getParam("nanColor")
        self.model().setColorForAllIdcs(hexNanColor)
        self.parent().selectionChanged.emit()

    def colorChangedFromMenu(self,event=None, hexColor = ""):
        ""
        rowIndex = self.rightClickedRowIndex
        dataIndex = self.model().getDataIndex(rowIndex)
        self.model().setColor(dataIndex,hexColor)
        self.parent().selectionChanged.emit()
        self.model().rowDataChanged(rowIndex)

    def selectSingleColor(self, selectedItemsOnly = False):
        "Select a single color for all items in the table"
        color = QColorDialog(parent=self.parent()).getColor()
        if color.isValid():
            if selectedItemsOnly:
                indices = self.selectionModel().selectedRows()
                self.model().setColorSelectedIdcs(color.name(),indices)
            else:
                self.model().setColorForAllIdcs(color.name())
            self.parent().selectionChanged.emit()

    def createMenu(self):
        ""
        legendLocations = ["upper right","upper left","center left","center right","lower left","lower right"]
        
        if self.model() is not None and hasattr(self.model(),"isEditable") and self.model().isEditable: #if editable - add the option to choose color from palette
            menu = createSubMenu(None,["Subset by ..","Color from palette","Add Legend at ..","Add Legend at (-NaN Color) .."])
            colors = self.mC.colorManager.getNColorsByCurrentColorMap(8)
            # categoricalColumns = self.mC.data.getCategoricalColumns(dataID = self.mC.getDataID())
            # for categoricalColumn in categoricalColumns:
            #     menu["Color by unique values in .."].addAction(categoricalColumn, lambda categoricalColumn = categoricalColumn: self.colorByUniqueValuesInCategoricalColumn(categoricalColumn))
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
                
            menu["main"].addAction("Single color for selected items", lambda : self.selectSingleColor(True))
            menu["main"].addAction("Single color for all items", self.selectSingleColor)
        else:
            menu = createSubMenu(None,["Add Legend at ..","Add Legend at (-NaN Color) ..","Color Range (min/max)"])
            menu["Color Range (min/max)"].addAction("user defined", self.parent().setMinMaxByUser)
            menu["Color Range (min/max)"].addAction("raw data", self.parent().setMinMaxByDefault)

        for legendLoc in legendLocations:
            menu["Add Legend at .."].addAction(legendLoc,lambda lloc = legendLoc: self.parent().addLegendToGraph(legendKwargs = {"loc":lloc}))
            menu["Add Legend at (-NaN Color) .."].addAction(legendLoc,lambda lloc = legendLoc: self.parent().addLegendToGraph(ignoreNaN = True,legendKwargs = {"loc":lloc}))
        
        menu["main"].addAction("Hide/Show in Legend", self.model().toggleInLegend)
        
        menu["main"].addAction("Remove Color Encoding", self.parent().removeFromGraph)
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
        row = self.rowAt(int(event.position().y()))
        column = self.columnAt(int(event.position().x()))
        return self.model().index(row,column)
    
    def mousePressEvent(self,e):
        ""
        
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
                self.menu.exec(QPoint(int(QCursor.pos().x()),int(QCursor.pos().y())) + QPoint(4,4))
                
                self.clickedRow = None 
            
            elif tableIndexCol == 1:
                #allow editing
                super().mouseReleaseEvent(e)   

        except Exception as e:
            print(e)


    def mouseMoveEvent(self,event):
        
        ""
        if self.state() == QAbstractItemView.State.EditingState:
            return 
        if not self.model().dataAvailable():
            return
        rowAtEvent = self.rowAt(int(event.position().y()))
        if rowAtEvent == -1:
            self.mouseOverItem = None
        else:
            self.mouseOverItem = rowAtEvent
            
            #self.parent().highlightColorGroup()

        self.model().completeDataChanged()
 
    def resizeColumns(self):
        ""
        columnWidths = [(0,20),(1,200)]
        for columnId,width in columnWidths:
            self.setColumnWidth(columnId,width)


        

