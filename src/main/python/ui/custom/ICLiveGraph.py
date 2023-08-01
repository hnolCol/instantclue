from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import * 

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib.lines import Line2D

from sklearn.preprocessing import scale

from ..utils import WIDGET_HOVER_COLOR, INSTANT_CLUE_BLUE, createMenu, getMessageProps
from .Widgets.ICButtonDesgins import ResetButton, BigArrowButton, PushHoverButton

from collections import OrderedDict

import numpy as np 
import pandas as pd 

TOOLTIP_STR = "Drag & drop numerical columns to view extra data.\nWorks in combination with hierarchical clustering and scatter plots."

class LiveGraph(QWidget):
    def __init__(self, parent=None, mainController=None, acceptedDragTypes = ["Numeric Floats"], zScoreNorm = False):
        super(LiveGraph, self).__init__(parent)
        self.setAcceptDrops(True)
        self.acceptedDragTypes = acceptedDragTypes
        self.acceptDrop = False
        self.zScoreNorm = zScoreNorm 
        
        #sourceData = mainController.data

        self.mC = mainController
        
        #print(self.dataSelectionHandler.getSelectedColumns)
        self.startActionOnThread = mainController.sendRequestToThread

        self._control()
        self._layout()
        self._connectEvents()

        self.setToolTip(TOOLTIP_STR)

    def _control(self):
        "control widgets"
        # a figure instance to plot on
        self.figure = Figure()
        self.liveGraph = BlitingLiveGraph(self, self.figure, mainController = self.mC)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # this is the Navigation widget
        # it takes the Canvas widget and a paren

        self.saveButton = BigArrowButton(buttonSize=(15,15), tooltipStr="Save chart as pdf.")
        self.resetButton = ResetButton(tooltipStr="Reset live graph.")
        self.plotTypeButton = PushHoverButton(text = "...", tooltipStr="Change plot type between bar and line.")
        self.plotTypeButton.setFixedSize(15,15)
        self.zScoreButton = PushHoverButton(text = "Z", tooltipStr="Show Z-Score")
        self.zScoreButton.setFixedSize(15,15)


    def _layout(self):
        "add layout"
        layout = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.zScoreButton)
        hbox.addWidget(self.plotTypeButton)
        hbox.addWidget(self.saveButton)
        hbox.addWidget(self.resetButton)
        hbox.setSpacing(1)
        hbox.setContentsMargins(2,0,2,0)
        
        layout.addLayout(hbox)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.layout().setContentsMargins(20,0,0,0)
        self.layout().setSpacing(2)

    def _connectEvents(self):
        ""
        self.plotTypeButton.clicked.connect(self.choosePlotType)
        self.resetButton.clicked.connect(self.clearGraph)
        self.saveButton.clicked.connect(self.saveGraph)
        self.zScoreButton.clicked.connect(self.toggleZScoreNorm)
    

    def choosePlotType(self,e = None):
        ""
        senderGeom = self.sender().geometry()
        topLeft = self.mapToGlobal(senderGeom.bottomLeft())
        menu = createMenu()
        menu.addAction("Bar")
        menu.addAction("Line")
        menu.addAction("Boxplot")
        self.sender().mouseOver = False
        action = menu.exec(topLeft)

        if action is not None:
            plotType = action.text() 
            self.liveGraph.setPlotType(plotType)
    
    def clearGraph(self,e = None):
        ""
        self.liveGraph.resetGraph()
    
    def saveGraph(self,e=None):
        ""
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Graph","","PDF Files (*.pdf);;PNG Files (*.png)")
        if fileName:
            self.liveGraph.figure.savefig(fileName)
            self.mC.sendMessageRequest(getMessageProps("Saved..","File {} has been saved.".format(fileName)))
            
    def toggleZScoreNorm(self,e=None):
        ""
        if self.zScoreButton.txtColor != INSTANT_CLUE_BLUE:
            self.zScoreButton.setTxtColor(INSTANT_CLUE_BLUE)
            self.zScoreNorm = True
        else:
            self.zScoreButton.setTxtColor("black")
            self.zScoreNorm = False
        
        if self.hasData():
            if not self.zScoreNorm:
                data = self.data
            else:
                data = pd.DataFrame(scale(self.data.values,axis=1),index=self.data.index,columns=self.data.columns)
                
            self.addDataToLiveGraph(data,False)

    def dragEnterEvent(self,e):
        ""
        
        dragType = self.getDragType()
        #check if type is accpeted and check if not all items are anyway already there
        if dragType in self.acceptedDragTypes:
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

    def updateGraphByIndex(self,dataIndex):
        "Update Graph"
        if not self.liveGraph.getFreezeState():
            self.liveGraph.setDataIndex(dataIndex)

    def dropEvent(self,e):
        ""
        e.accept()
        self.zScoreNorm = False #reset z-score 
        self.zScoreButton.setTxtColor("black")
        dataID = self.mC.mainFrames["data"].getDataID()
        columnNames = self.getDragColumns()
        data = self.mC.data.getDataByColumnNames(dataID = dataID, columnNames = columnNames)["fnKwargs"]["data"]
        self.addDataToLiveGraph(data)
    
    def addDataToLiveGraph(self,data,storeData = True):
        #try to find grouping
        colorMapper = self.mC.grouping.getColorsForGroupMembers()
        #init live graph
        self.liveGraph.dataChanged.emit(data,{} if colorMapper is None else colorMapper)
        if storeData:
            self.data = data

    def getFreezeState(self):
        ""
        return self.liveGraph.getFreezeState()
    
    def setFreezeState(self, state : bool) -> None:
        ""
        self.liveGraph.setFreezeState(state)

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

    def hasData(self):
        ""
        return not self.liveGraph.data.empty

    def resizeEvent(self,event=None):
        ""
        if not self.liveGraph.getResizeTrigger():
            self.liveGraph.setResizeTrigger(True)
        #self.liveGraph.canvasResized()

    def enterEvent(self,event=None):
        ""
        if self.liveGraph.getResizeTrigger():
            self.liveGraph.canvasResized()

    def leaveEvent(self,event=None):
        ""
        if self.liveGraph.getResizeTrigger():
            self.liveGraph.canvasResized()

class BlitingLiveGraph(QWidget):
    dataChanged = pyqtSignal(pd.DataFrame,dict)
    freezeImageChanged = pyqtSignal(bool)

    def __init__(self,parent, figure, plotType = "Line", mainController = None):
        ""
        super(BlitingLiveGraph, self).__init__(parent)
        self.figure = figure
        self.plotType = plotType
        self.data = pd.DataFrame() 
        self.selectionIndex = None
        self.frozen = False
        self.resized = False
        self.colorMapper = {}
        self.mC = mainController
        self.dataChanged.connect(self.setData)
        self.freezeImageChanged.connect(self.setFreezeState)
        
    def addAxis(self):
        if not hasattr(self,"ax"):

            self.ax = self.figure.add_subplot(111)
            self.figure.subplots_adjust(left=0.075,right=0.95, top = 0.965, bottom = 0.2)
        else:
            self.ax.clear()
            
    def addArtist(self):
        ""
        if self.plotType == "Line" and not hasattr(self,"hoverLine"):

            self.addLine()
        
        elif self.plotType == "Bar" and not hasattr(self,"hoverBars"):

            self.addBars()

        elif self.plotType == "Boxplot" and not hasattr(self,"hoverBoxplots"):

            self.addBoxplots()

    def addBars(self):
        ""

        self.hoverBars = self.ax.bar(x = self.xTicks - 0.4, 
                                    height = [0]*self.xTicks.size, 
                                    width = 0.8,
                                    align = "center",
                                    color = INSTANT_CLUE_BLUE,
                                    edgecolor = "black",
                                    linewidth = 0.5,
                                    alpha = 0.8)

    def addBoxplots(self):
        ""            
        self.hoverBoxplotLines =  [Line2D(xdata = [], 
                                            ydata = [],
                                            linewidth = 0.5,
                                            color= "black",
                                            linestyle="-",
                                            zorder = 1e5
                                            ) for _ in self.xTicks]
        
        for l in self.hoverBoxplotLines:
            self.ax.add_line(l)

        self.hoverBoxplots = self.ax.bar(x = self.xTicks - 0.4, 
                                height = [0]*self.xTicks.size, 
                                width = 0.8,
                                align = "center",
                                color = ["white"],
                                edgecolor = "black",
                                linewidth = 0.8,
                                alpha = 0.85,
                                zorder = 2e5)

    def addLine(self):
        ""
        #crate line
        self.hoverLine = self.ax.plot([],[], 
                                        marker = "o", 
                                        color = "black", 
                                        linewidth = 0.8, 
                                        markerfacecolor = "white", 
                                        markeredgecolor="black")
        #hide line
        self.setInvisible()

    def addText(self):
        ""
        self.hoverText = self.ax.text(0.1, 0.99, "",
            horizontalalignment='left',
            verticalalignment='top',
            transform=self.ax.transAxes)

    def addXTicks(self, xTicksVisible, numColumn):
        #add some minor border
        if self.plotType == "Line":
            self.ax.set_xticks(xTicksVisible)
            self.ax.set_xlim(-0.25,numColumn-0.75)
        elif self.plotType == "Bar":
            self.ax.set_xticks(xTicksVisible+0.5)
            self.ax.set_xlim(-0.5,numColumn + 0.5)
        elif self.plotType == "Boxplot":
            self.ax.set_xticks(xTicksVisible+0.5)
            self.ax.set_xlim(-0.5,numColumn + 0.5)
        
        self.ax.set_xticklabels(self.xtickLabels,rotation = 45, ha="right")

    def adjustXTicksToSize(self, numColumn):
        ""
        figSize = self.figure.get_size_inches() * self.figure.get_dpi()
        fontSizeLabels = self.mC.config.getParam("xtick.labelsize")
        labelsToFit = int(figSize[0] / (fontSizeLabels * 3)) 
        #dynamic adaptions of xticks
        if numColumn > labelsToFit:
            self.xTicks = np.arange(numColumn)
            xTicksVisible = np.array([int(x) for x in np.linspace(0,numColumn-1,num = labelsToFit)])
            self.xtickLabels = [self.data.columns[x] for x in xTicksVisible]
        else:
            self.xTicks = np.arange(numColumn)
            self.xtickLabels = self.data.columns
            xTicksVisible = self.xTicks
        
        self.addXTicks(xTicksVisible,numColumn)

    def canvasResized(self,event=None):
        "Handle background upgrade upon figure resize"
        if hasattr(self,"ax") and hasattr(self,"data") and self.data.columns.size > 0:
            #adjust xticks
            self.adjustXTicksToSize(numColumn=self.data.columns.size)
            #update background
            self.updateBackground(redraw=True)
            #reset trgígger
            self.setResizeTrigger(False)

    def updateAxis(self, redraw=True):
        '''
        Update artists using blit.
        '''
        if self.getResizeTrigger():
            self.canvasResized()

        if not hasattr(self,'background'):
            self.background =  self.figure.canvas.copy_from_bbox(self.ax.bbox)
                
        self.figure.canvas.restore_region(self.background)
        if self.plotType == "Line":
            self.ax.draw_artist(self.hoverLine[0])	
        elif self.plotType == "Bar":
            for bar in self.hoverBars:
                self.ax.draw_artist(bar)
        elif self.plotType == "Boxplot":
            for l in self.hoverBoxplotLines:
                self.ax.draw_artist(l)
            for box in self.hoverBoxplots:
                self.ax.draw_artist(box)
        self.ax.draw_artist(self.hoverText)
        self.figure.canvas.blit(self.ax.bbox)

    def clearAxis(self):
        ""
        if hasattr(self,"ax"):
            self.ax.clear() 
    
    def resetGraph(self):
        ""
        self.data = pd.DataFrame() 
        self.selectionIndex = None
        self.removeAxis()
        self.redraw()

    def setFreezeState(self,state : bool) -> None:
        ""
        setattr(self,"frozen",state)
        if self.selectionIndex is not None:
        
            self.updateNumber(len(self.selectionIndex))
            self.updateAxis()

    def getFreezeState(self) -> bool:
        ""
        return getattr(self,"frozen")

    def setXAxisProps(self) -> None:
        ""
        numColumn = self.data.columns.size
        try:
            #save columnName per idx
            self.xTickLabels = OrderedDict([(n,columnName) for n,columnName in enumerate(self.data.columns)])
            self.adjustXTicksToSize(numColumn)
            
        except Exception as e:
            print(e)

    def setYAxisProps(self):
        ""
        #calculate quants 
        self.quantiles = np.nanquantile(self.data.values,[0,0.25,0.5,0.75,1],axis=0)
        minValue = np.min(self.quantiles[0,:])
        maxValue = np.max(self.quantiles[4,:])#4 quantile calcs, e.g. max = 1

        self.ax.set_ylim(minValue,maxValue)

       
    def setData(self, data : pd.DataFrame, colorMapper : dict):
        ""
        self.freezeImageChanged.emit(False)
        if data is not None and isinstance(data,pd.DataFrame):
            if data.index.size == 0:
                return
            self.data = data.dropna(axis=1,how="all") #remove columns with only nan
            if self.data.empty:
                return
            self.removeArtists()
            self.addAxis() 
            self.setXAxisProps()
            self.setYAxisProps()
            self.addArtist()
            self.addText()
            self.setLineProps()
        try:
            self.plotDataQauntiles()
        except Exception as e:
            print(e)
        self.updateBackground()
        self.groupColors = [colorMapper[columnName] if colorMapper is not None and columnName in colorMapper else "white" for columnName in self.data.columns ] #
        self.colorMapper = colorMapper

    def setInvisible(self,event = None):
        '''
        '''
        if self.plotType == "Line" and self.hoverLine[0].get_visible():
            self.hoverLine[0].set_visible(False)
        elif self.plotType == "Bar":
            for bar in self.hoverBars:
                bar.set_visible(False)   
        elif self.plotType == "Boxplot":
            for bar in self.hoverBoxplots:
                bar.set_visible(False)   
            for l in self.hoverBoxplotLines:
                l.set_visible(False)
        if hasattr(self,"hoverText"):
            self.hoverText.set_visible(False)

    def setLineProps(self):
        ""
        if hasattr(self,"hoverLine"):
            if self.data.columns.size > 30:
                self.hoverLine[0].set_marker("None")
            else:
                self.hoverLine[0].set_marker("o")

    def setPlotType(self,plotType):
        ""
        if plotType in ["Line","Bar","Boxplot"]:
            self.plotType = plotType
            self.dataChanged.emit(self.data,self.colorMapper)

    def setResizeTrigger(self,resized):
        ""
        self.resized = resized

    def getResizeTrigger(self):
        ""
        return self.resized

    def updateGraph(self):
        ""
        # if self.selectionIndex is not None:
        #     #plot lines

        #     pass
        
        if self.plotType == "Line":
            Y = self.data.loc[self.selectionIndex,:].mean().values
            self.updateLineData(self.xTicks, Y)
        elif self.plotType == "Bar":
            Y = self.data.loc[self.selectionIndex,:].mean().values
            self.updateBarData(self.xTicks, Y)
        elif self.plotType == "Boxplot":
            Y = self.data.loc[self.selectionIndex,:].quantile(q=[0.0,0.25,0.5,0.75,1.0]).values
            self.updateBoxData(self.xTicks, Y)
        numberRows = len(self.selectionIndex)
        self.updateNumber(numberRows)
        self.updateAxis()

    def updateNumber(self,n):
        ""
        
        self.hoverText.set_visible(True)
        self.hoverText.set_text(f"n : {n}" if not self.frozen else f"n: {n} - frozen")

    def updateBarData(self,x,y):
        ""
        if not hasattr(self, "groupColors"):
            self.groupColors = [INSTANT_CLUE_BLUE] * len(self.hoverBars)
        if x.size == y.size and x.size == len(self.hoverBars):
            for n,bar in enumerate(self.hoverBars):
                bar.set_visible(True)
                bar.set_x(x[n])
                bar.set_height(y[n])
                bar.set_fc(self.groupColors[n])

    def updateBoxData(self,x,y):
        ""
        if not hasattr(self, "groupColors"):
            self.groupColors = ["white"] * len(self.hoverBoxplots)
        if x.size == y.shape[1] and x.size == len(self.hoverBoxplots):
            for n,bar in enumerate(self.hoverBoxplots):
                if np.any(np.isnan(y[:,n])):
                    bar.set_visible(False)
                    continue
                bar.set_visible(True)
                bar.set_x(x[n])
                bar.set_height(y[3,n]-y[1,n]) #height for box is q1 - q2
                bar.set_y(y[1,n])
                bar.set_fc(self.groupColors[n])

                self.hoverBoxplotLines[n].set_visible(True)
                self.hoverBoxplotLines[n].set_data(
                                [x[n]+0.4,x[n]+0.4], #xdata
                                [y[0,n],y[-1,n]]) #ydata


    def updateLineData(self,x,y,size=None):
        ""
        self.hoverLine[0].set_visible(True)
        self.hoverLine[0].set_data(x,y)
        

    def updateBackground(self, redraw = True):
        ""
        if not hasattr(self,"figure"):
            return
        if self.data.empty:
            return
        if redraw:
            if hasattr(self,'background'):
                self.setInvisible()
            self.redraw()
        self.background =  self.figure.canvas.copy_from_bbox(self.ax.bbox)

    def plotDataQauntiles(self):
        ""
        if hasattr(self,"quantiles"):
            if self.plotType == "Line":
                x = self.xTicks
                q0 = self.quantiles[0,:]
                q25 = self.quantiles[1,:]
                q50 = self.quantiles[2,:]
                q75 = self.quantiles[3,:]
                q100 = self.quantiles[4,:]
            elif self.plotType in ["Bar","Boxplot"]:
                x = np.linspace(0,self.xTicks.size-0.2,num=self.xTicks.size * 2)
                q0 = np.repeat(self.quantiles[0,:],2)
                q25 = np.repeat(self.quantiles[1,:],2)
                q50 = np.repeat(self.quantiles[2,:],2)
                q75 = np.repeat(self.quantiles[3,:],2)
                q100 = np.repeat(self.quantiles[4,:],2)
            #plot min and max quantiles
            self.ax.fill_between(x,q0,q100,alpha=0.3,color="lightgrey",linewidth=0.1)
            #plot 25 and 75 quantiles
            self.ax.fill_between(x,q25,q75,alpha=0.7,color="grey",linewidth=0.1)
            #plot median
            self.ax.plot(x,q50,linestyle="--",color="black",linewidth=0.5)
            self.redraw()

    def setDataIndex(self,dataIndex):
        ""
        
        if dataIndex is None:
            return

        if not isinstance(dataIndex,pd.Series):
            dataIndex = pd.Series(dataIndex)

        if not self.data.empty:

            idxChecked = [idx for idx in dataIndex.values if idx in self.data.index]
            self.selectionIndex = idxChecked
            self.updateGraph()
        
    def redraw(self):
        ""
        if hasattr(self,"figure"):
            self.figure.canvas.draw()
    
    def removeArtists(self):
        ""
        if hasattr(self,"hoverLine"):
            del self.hoverLine
        if hasattr(self,"hoverBars"):
            del self.hoverBars

    def removeAxis(self):
        ""
        if hasattr(self,"ax"):
            self.ax.remove()
            del self.ax