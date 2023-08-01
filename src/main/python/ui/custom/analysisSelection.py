from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from .ICCollapsableFrames import CollapsableFrames
from .Widgets.ICButtonDesgins import DataHeaderButton, ViewHideIcon, FindReplaceButton, ResetButton, BigArrowButton
from .ICDataTreeView import DataTreeView
from ..dialogs.ICDataInputDialog import ICDataInput
from ..utils import WIDGET_HOVER_COLOR, getHoverColor, INSTANT_CLUE_BLUE, getCollapsableButtonBG
import pandas as pd
from collections import OrderedDict

from .utils import INSTANT_CLUE_ANAYLSIS


class AnalysisSelection(QWidget):
    
    def __init__(self, parent=None, mainController = None):
        super(AnalysisSelection, self).__init__(parent)
        
        self.mC = mainController
        self.__controls()
        self.__layout() 
        self.__connectEvents()
        

    def __controls(self):
        # set up collapsable frame widget
        # we need extra frame to get parent size correctly
        self.dataTreeFrame = QFrame(self)
        #add widget to frame
        self.frames = CollapsableFrames(parent=self.dataTreeFrame,buttonDesign=DataHeaderButton,spacing=0)
        frameWidgets = []
        self.dataHeaders = dict()
        for hierAnalysisSteps in INSTANT_CLUE_ANAYLSIS:
            #hierAnalysisSteps is a single entry dict
            for k, v in hierAnalysisSteps.items():
                header = k
                values = v
            self.dataHeaders[header] = DataTreeView(self.frames, mainController = self.mC, tableID = header)
            frame = {"title":header,
                     "open":False,
                     "fixedHeight":False,
                     "height":0,
                     "layout":self.dataHeaders[header].layout()}
            frameWidgets.append(frame)
            self.dataHeaders[header].addData(pd.Series(values))
            self.dataHeaders[header].hideShowShortCuts()
            self.dataHeaders[header].table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.frames.addCollapsableFrame(frameWidgets, 
                                        closeColor = getCollapsableButtonBG(),
                                        openColor = getCollapsableButtonBG(),
                                        dotColor = INSTANT_CLUE_BLUE,
                                        hoverColor = getHoverColor(),
                                        hoverDotColor = WIDGET_HOVER_COLOR, 
                                        widgetHeight = 20)

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.dataTreeFrame)
        self.dataTreeFrame.setLayout(QVBoxLayout())
        self.dataTreeFrame.layout().setContentsMargins(0,0,0,0)
        self.dataTreeFrame.layout().setSpacing(0)
        self.dataTreeFrame.layout().addWidget(self.frames)

        self.layout().setContentsMargins(0,0,0,0)
        #self.layout().setSpacing(1)

    def __connectEvents(self):
        ""

    def getDragTask(self):
        ""
        if hasattr(self,"draggedTask"):
            return self.draggedTask 
        
    def getTaskType(self):
        if hasattr(self,"taskType"):
            return self.taskType
    
    def getDataID(self):
        ""
        if hasattr(self,"dataID"):
            return self.dataID 

    def updateDragData(self, draggedTask, taskType):
        """
        Dragged Columsn and dragType is stored and can be accesed by 
        function getDragColumns and getDragType (or directly)
        """
        self.draggedTask = draggedTask
        self.taskType = taskType

    
    def runAnalysis(self):
        """
        Maybe To-Do: 
            Add IDs instead of string to find required function? Double nameed tasks are useless anyway?
        """
        try:
            exists, graph = self.mC.getGraph()
            task = self.draggedTask.values[0]
            if task in ["Line (slope=1)","Line (y = m*x + b)"]:
                
                if exists:
                    if task == "Line (y = m*x + b)":
                        askLineData = ICDataInput(mainController=self.mC, title = "Provide line's slope and intercep.\ny = m*x + b",valueNames = ["m","b"], valueTypes = {"m":float,"b":float})
                        if askLineData.exec():
                            m, b = askLineData.providedValues["m"], askLineData.providedValues["b"]
                        else:
                            return
                    elif task == "Line (slope=1)":
                        m, b = 1, 0

                    graph.addLinearLine(m,b) #default values are m = 1, b = 0
                    graph.updateFigure.emit()
            elif task == "Cross":
                askCrossData = ICDataInput(mainController=self.mC, title = "Provide the x- and y-axis coordinates for the cross.",valueNames = ["x","y"], valueTypes = {"x":float,"y":float})
                if askCrossData.exec():
                        xCrossCoord, yCrossCoord = askCrossData.providedValues["x"], askCrossData.providedValues["y"]
                        graph.addCrossLine(xCrossCoord, yCrossCoord)
                        graph.updateFigure.emit() 
            
            elif task == "Vertical Line":
                askVLineData = ICDataInput(mainController=self.mC, title = "Provide the x-axis coordinate for the line.",valueNames = ["x"], valueTypes = {"x":float})
                if askVLineData.exec():
                    xCoord = askVLineData.providedValues["x"]
                    graph.addVerticalLine(xCoord)
                    graph.updateFigure.emit() 
            elif task == "Horizontal Line":
                askHLineData = ICDataInput(mainController=self.mC, title = "Provide the y-axis coordinate for the line.",valueNames = ["y"], valueTypes = {"y":float})
                if askHLineData.exec():
                    xCoord = askHLineData.providedValues["y"]
                    graph.addHorizontalLine(xCoord)
                    graph.updateFigure.emit() 

            elif task == "Quadrant Lines":
                reqFloats = ["x_min","x_max","y_min","y_max"]
                askQuadData = ICDataInput(mainController=self.mC,title = "Provide x and y axis quadrant limits",valueNames = reqFloats, 
                        valueTypes = {"x_min":float,"x_max":float,"y_min":float,"y_max":float})
                if askQuadData.exec():
                    quadrantCoords = [askQuadData.providedValues[floatName] for floatName in reqFloats]
                    graph.addQuadrantLines(quadrantCoords)
                    
                else:
                    return
            elif task == "Axes Diagonal":

                if exists:
                    graph.addDiagonal()
                    graph.updateFigure.emit()

            elif task in ["Line from file","Line from clipboard"]:

                if exists:
                    if task == "Line from clipboard":
                        try:
                            data = pd.read_clipboard()
                        except:
                            self.mC.sendMessageRequest({"title":"Error..","message":"No suitable data found in clipboard"})
                    elif task == "Line from file":
                        dlg = QFileDialog(caption="Select File for Line data",
                                filter = "ICLoad Files (*.txt *.csv *tsv );;Text files (*.txt *.csv *tsv)")
                        dlg.setFileMode(QFileDialog.ExistingFiles)
                        if dlg.exec():
                            fileName = dlg.selectedFiles()[0]
                            data = pd.read_csv(fileName,sep="\t")
                        else:
                            return 

                    if data.shape[1] == 2 and data.shape[0] > 1:

                        x = data.values[:,0]
                        y = data.values[:,1]

                        graph.addLineByArray(x,y)
                    
                    else:
                        
                        self.mC.sendMessageRequest({"title":"Error","message":"Data must have two columns and more than 1 row. Found data shape: {}:{}\nColumn headers are expected.".format(data.index.size,data.columns.size)})

            elif task in ["linear fit","lowess"]:
               
                if exists and graph.hasScatters():
                    
                    columnPairs = graph.data["columnPairs"]
                    dataID = self.mC.getDataID()
                    funcProps = {"key":"plotter:addLowessFit" if task == "lowess" else "plotter:addLinearFit",
                                "kwargs":{"dataID":dataID,"numericColumnPairs":columnPairs}}
                    self.mC.sendRequestToThread(funcProps)

            elif task in ["t-test","Welch-test","Wilcoxon","(Whitney-Mann) U-test","One-sample t-test","Wilcoxon signed-rank test"]:

                exists, graph = self.mC.getGraph() 
                
                if exists and graph.isChartCompatibleWithInteractiveStats():
                    graph.enableInteractiveStats()
                    
        except Exception as e:
            print(e)


    def updateDataInTreeView(self,columnNamesByType):
        """Add data to the data treeview"""
        if isinstance(columnNamesByType,dict):
            for headerName, values in columnNamesByType.items():
                if headerName in self.dataHeaders:
                    if isinstance(values,pd.Series):
                        self.dataHeaders[headerName].addData(values)    
                    else:
                        raise ValueError("Provided Data are not a pandas Series!") 
    
    def updateDataIDInTreeViews(self):
        "Update Data in Treeview:: settingData"
        for treeView in self.dataHeaders.values():
            treeView.setDataID(self.dataID)


    def updateColumnState(self,columnNames, newState = False):
        "The column state indicates if the column is used in the graph or not (bool)"
        for dataHeader, treeView in self.dataHeaders.items():
            treeView.setColumnState(columnNames,newState)
  
    def sendToThread(self, funcProps, addSelectionOfAllDataTypes = False, addDataID = False):
        ""
        
        if addSelectionOfAllDataTypes:
            funcProps = self.addSelectionOfAllDataTypes(funcProps)
        if addDataID and "kwargs" in funcProps:
            funcProps["kwargs"]["dataID"] = self.dataID
        self.mC.sendRequestToThread(funcProps)
              