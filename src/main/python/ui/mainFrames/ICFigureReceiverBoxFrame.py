from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, plot
from collections import OrderedDict
from ..custom.ICReceiverBox import ReceiverBox
#from ..plotter.plotter import Plotter
from ..plotter.plotManager import ICPlotter
import numpy as np 


class MatplotlibFigure(QWidget):
    def __init__(self, parent=None, mainController=None):
        super(MatplotlibFigure, self).__init__(parent)
        self.setAcceptDrops(True)
        self.acceptDrop = False
        self.isPlotting = False

        self.mC = mainController
        
        self.startActionOnThread = mainController.sendRequestToThread
        # a figure instance to plot on
        self.figure = figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        #self.plotter = Plotter(self,sourceData ,self.figure)
        self.ICPlotter = ICPlotter(self.mC,self.figure)
        #self.ICPlotter.graph.setData()

        self.receiverBoxes = OrderedDict() 
        self.receiverBoxes["Numeric Floats"] = ReceiverBox(parent=self)
        self.receiverBoxes["Categories"] = ReceiverBox(parent=self, title="Categories", acceptedDragTypes = ["Integers","Categories"])
        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.receiverBoxes["Numeric Floats"])
        layout.addWidget(self.receiverBoxes["Categories"])
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        self.layout().setContentsMargins(0,0,0,0)


    def initiateChart(self, *args, **kwargs):
        ""
       
        try:
            self.clearFigure(forceRedraw=False)
            plotType = self.mC.mainFrames["right"].getCurrentPlotType()
            if "dataProps" not in kwargs:
                kwargs["dataProps"] = self.getDataProps()
            #kwargs
            kwargs["colorMap"] = self.mC.data.colorManager.colorMap
            kwargs["selectedPlotType"] = plotType
            
            funcKey = kwargs["dataProps"]["funcProps"]
            funcKey["key"] = "plotter:getData"
            funcKey["kwargs"]["plotType"] = plotType
            funcKey["kwargs"]["figureSize"] = self.mC.getFigureSize()
            
            
            self.mC.sendRequestToThread(funcKey)
            
           # self.plotter.initiateChart(*args,**kwargs)#
        except Exception as e:
            print(e)

    def restoreGraph(self,plotType,graphData):
        ""
        if self.setGraph(plotType):
            self.setData(graphData)


    def getCanvas(self):
        ""
        if hasattr(self,"canvas"):
            return self.canvas

    def getFigure(self):
        ""
        if hasattr(self,"figure"):
            return self.figure

    def getSelectedColumns(self,dataType="all"):
        ""

        if not hasattr(self, "dataSelectionHandler"):
            self.dataSelectionHandler = self.mC.mainFrames["data"]
        selectedColumns = self.dataSelectionHandler.getSelectedColumns(dataType = dataType)

        return selectedColumns

    def getDragType(self):
        ""
        if not hasattr(self, "dataSelectionHandler"):
            self.dataSelectionHandler = self.mC.mainFrames["data"]
        return self.dataSelectionHandler.getDragType()

    def getDragColumns(self):
        ""
        if not hasattr(self, "dataSelectionHandler"):
            self.dataSelectionHandler = self.mC.mainFrames["data"]
        return self.dataSelectionHandler.getDragColumns()
    
    def getReceiverBoxItems(self):
        ""
        items = dict()
        self.addReceiverBoxItems(items)
        return items
    
    def updateReceiverBoxItemsSilently(self, receiverBoxItems):
        ""
        for receiverBoxName, receiverBox in self.receiverBoxes.items():
            columnAlias = "numericColumns" if receiverBoxName == "Numeric Floats" else "categoricalColumns"
            if columnAlias in receiverBoxItems:
                receiverBox.addItems(receiverBoxItems[columnAlias])

    def getToolbarState(self):
        "Careful, works only if matplitlib > 3.3, otherwise active"
        if hasattr(self.toolbar,"_active"):
            return self.toolbar._active
        else:
            if hasattr(self.toolbar, "mode"):
                if self.toolbar.mode == "": #match _active outcome
                    return None
                else:
                    return self.toolbar.mode
            
    
    def setData(self,data):
        ""
        if hasattr(self.ICPlotter,"graph"):
            self.ICPlotter.graph.setData(data)

    def setGraph(self,plotType):
        ""
        
        if hasattr(self,"ICPlotter"):
            return self.ICPlotter.setGraph(plotType)  
        else:
            return False

    def refreshQuickSelectSelection(self):
        ""
        response = self.mC.mainFrames["data"].qS.updateDataSelection()
        if response is None:
            self.updateFigure()

    def getDataProps(self):
        ""
        dataProps = {}
        dataProps["requestData"] = self.startActionOnThread
        dataProps = self.addReceiverBoxItems(dataProps)
        dataID = self.mC.mainFrames["data"].getDataID()
        
        dataProps["funcProps"] = {"key":"data::getDataByColumnNamesForPlotter",
                    "kwargs":{"dataID":dataID,
                    "numericColumns":dataProps["numericColumns"] , "categoricalColumns" : dataProps["categoricalColumns"]
                    }}
        return dataProps

    def addReceiverBoxItems(self, dataProps):
        ""
        if isinstance(dataProps,dict):
            dataProps["numericColumns"] = self.receiverBoxes["Numeric Floats"].getItemNames()
            dataProps["categoricalColumns"] = self.receiverBoxes["Categories"].getItemNames()
        
        return dataProps

    def getReceiverBoxes(self):
        ""
        return self.receiverBoxes.values()

    def renameColumns(self,columnNameMapper):
        ""
        if not isinstance(columnNameMapper,dict):
            return
            #raise ValueError("columnNameMapper must be a dict!")
        try:
            for itemName, newItemName in columnNameMapper.items():
                for receiverBox in self.getReceiverBoxes():
                    receiverBox.renameItem(itemName,newItemName)
            self.updateReceiverBoxItems() 
        except Exception as e:
            print(e)
            
    def setCategoryIndexMatch(self,categoryIndexMatch,categoryEncoded="color"):
        ""
        
        exists, graph = self.mC.getGraph()
        if exists:
            if categoryEncoded == "QuickSelect":
                graph.setQuickSelectCategoryIndexMatch(categoryIndexMatch)
            elif categoryEncoded == "color":
                graph.setColorCategoryIndexMatch(categoryIndexMatch)
            elif categoryEncoded == "size":
                graph.setSizeCategoryIndexMatch(categoryIndexMatch)

    def updateHclustSize(self,sizeData):
        ""
        exists, graph = self.mC.getGraph()
        if exists:
            graph.updateHclustSize(sizeData)

    def updateHclustColor(self, colorData, colorGroupData, cmap = None, title=""):
        ""
        exists, graph = self.mC.getGraph()
        if exists:
            graph.updateHclustColor(colorData,colorGroupData,cmap,title)

    def updateDataInPlotter(self):
        ""
        self.ICPlotter.graph.updateData()

    def updateScatterProps(self,propsData):
        ""
        exists, graph = self.mC.getGraph()
        if exists:
            graph.updateScatterProps(propsData)
            graph.updateFigure.emit()

    def updateQuickSelectSelectionInGraph(self,propsData):
        ""
        
        exists, graph = self.mC.getGraph()
        if exists:
           
            if hasattr(graph,"updateQuickSelectItems"):#graph.isHclust() or graph.isLinePlot() or graph.isBoxplotViolinBar():
                
                graph.updateQuickSelectItems(propsData)
                
                graph.updateFigure.emit()
        

    def updateReceiverBoxItems(self):
        ""
        for receiverBox in self.getReceiverBoxes():
            receiverBox.updateItems()

    def resetReceiverBoxes(self):
        "Reset receiver boxes"
        for receiverBox in self.getReceiverBoxes():
            receiverBox.clearDroppedItems(reportStateToTreeView=False,emitReceiverBoxChangeSignal=False)
        self.recieverBoxItemsChanged()

    def addItemsToReceiverBox(self, columnNamesByType, numUniqueValues = None):
        "Add Items to Receiver box (if they are not in there)"
        for dataType, columnNames in columnNamesByType.items():   
            recieverBox = self.receiverBoxes[dataType]
            if dataType in self.receiverBoxes and len(columnNames) != 0:
                recieverBox.addItems(columnNames)
        self.recieverBoxItemsChanged()

    def removeItemsFromReceiverBox(self, columnNames):
        "Removes items from receiver boxes."
        if self.isAnyItemInAnyReceiverBox(columnNames):
            for recieverBox in self.getReceiverBoxes():
                recieverBox.removeItems(columnNames)
            self.recieverBoxItemsChanged()

    def setColumnStateInTreeView(self, columnNames, newState = False):
        "Changes the state (in graph state) of a column"
        self.mC.mainFrames["data"].dataTreeView.updateColumnState(columnNames,newState)

    def isAnyItemInAnyReceiverBox(self,items):
        ""
        checkNum = any(itemName in self.receiverBoxes["Numeric Floats"].items for itemName in items)
        if checkNum: 
            return True
        checkCat = any(itemName in self.receiverBoxes["Categories"].items for itemName in items)
        if checkCat:
            return True
        else:
            return False

    def areReceiverBoxesEmpty(self):
        ""
        return len(self.receiverBoxes["Categories"].items) == 0 \
            and len(self.receiverBoxes["Numeric Floats"].items) == 0

    def recieverBoxItemsChanged(self, alreadyChecked = False, *args, **kwargs):
        "Signal that items have changed"
        
        if self.areReceiverBoxesEmpty():
            self.clearFigure(forceRedraw=True)
        else:
            if not alreadyChecked:
                self.mC.mainFrames["right"].checkType()
            if self.mC.mainFrames["right"].getCurrentPlotType() == "addSwarmplot":
                exists,graph = self.mC.getGraph()
                if exists:
                    graph.addSwarm(self.mC.mainFrames["data"].getDataID(),**self.addReceiverBoxItems({}))
                    self.mC.mainFrames["right"].setType(graph.plotType, update=False) # reset original plot type
            else:
                self.initiateChart(*args, **kwargs)

    def clearFigure(self, forceRedraw=False):
        ""
        self.ICPlotter.clearFigure()
        self.ICPlotter.redraw(forceRedraw=True)
        self.mC.resetGroupColorTable.emit()
        self.mC.resetGroupSizeTable.emit()
        self.mC.resetLabelTable.emit()
        self.mC.resetTooltipTable.emit()
        self.mC.resetStatisticTable.emit()
        self.mC.resetMarkerTable.emit()

    def updateFigure(self, newPlot = False,*args,**kwargs):
        "Update Figure (e.g. redraw)."
        exists,graph = self.mC.getGraph()
        if exists:
            graph.updateFigure.emit()

    def updatePlotData(self):
        ""
        self.recieverBoxItemsChanged()

    # def updateActivePlotterFn(self,fnName,fnKwargs, updatePlot = True):
    #     ""
    #     activePlotter = self.plotter.get_active_helper()
    #     if activePlotter is not None and hasattr(activePlotter,fnName):
    #         getattr(activePlotter,fnName)(**fnKwargs)
    #         if updatePlot:
    #             #check if a new plot was made by setting data
    #             #then update quick select items
    #             self.updateFigure()

    def addLine(self,lineData):
        ""
        exists,graph = self.mC.getGraph()
        if exists and graph.hasScatters():
            graph.addLines(lineData)
    
    def addArea(self,areaData):
        ""
        exists,graph = self.mC.getGraph()
        if exists and graph.hasScatters():
            graph.addArea(areaData)
    
    def addLineCollections(self,lineCollections):
        ""
        exists,graph = self.mC.getGraph()
        if exists:
            graph.addLineCollections(lineCollections)

    def addTooltip(self,tooltipColumnNames,dataID):
        ""
        print("BUM")
        print(tooltipColumnNames)
        if len(tooltipColumnNames) > 0:
            exists,graph = self.mC.getGraph()
            if exists:
                print(tooltipColumnNames)
                graph.addTooltip(tooltipColumnNames,dataID)


    def activateHoverInScatter(self):
        ""
        self.ignoreHoverInScatter(ignore=False)

    def resizeEvent(self,event=None):
        ""
        if hasattr(self.ICPlotter,"graph"):
            
            self.ICPlotter.graph.setResizeTrigger(True)
            
    def dropEvent(self,e = None):
        ""
        if self.acceptDrop:
            self.mC.mainFrames["data"].analysisSelection.runAnalysis()
            e.accept()
        
    def dragEnterEvent(self,e):
        ""
        #get potential sources (e.g. TableViews that hold analysis tasks)
        analysisSection = self.mC.mainFrames["data"].analysisSelection
        dataTreeViews = [v.table for v in analysisSection.dataHeaders.values()]
        
        if self.mC.isDataLoaded() and e.source() in dataTreeViews:
            self.acceptDrop = True
        else:
            self.acceptDrop = False
        
        e.accept()

    def dragMoveEvent(self, e):
        "Ignore/acccept drag Move Event"
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()

    def leaveEvent(self,event=None):
        ""
        exists,graph = self.mC.getGraph()
        if exists and graph.hasScatters():
            for scatterPlot in graph.scatterPlots.values():
                scatterPlot.setHoverObjectsInvisible(leftWidget=True)
        

    def setIsPlotting(self,isPlotting):
        ""
        setattr(self,"isPlotting",isPlotting)
    
    def setMask(self,maskIndex):
        ""
        if maskIndex is None:
            self.resetMask()
        else:
            exists, graph = self.mC.getGraph()
            if exists:
                graph.setMask(maskIndex)
            
       
    def resetMask(self):
        ""
        exists, graph = self.mC.getGraph()
        if exists:
            graph.resetMask()

    def handleMaskChange(self):
        ""
        if not self.areReceiverBoxesEmpty():
            self.recieverBoxItemsChanged(alreadyChecked=True) 
