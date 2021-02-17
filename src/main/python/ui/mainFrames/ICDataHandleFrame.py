
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..dialogs.ICLoadFileDialogs import PlainTextImporter, ExcelImporter
from ..dialogs.ICCategoricalFilter import CategoricalFilter
from ..custom.ICCollapsableFrames import CollapsableFrames
from ..custom.buttonDesigns import DataHeaderButton, CollapsButton
from ..custom.ICQuickSelect import QuickSelect
from ..custom.dataFrameSelection import CollapsableDataTreeView
from ..custom.ICDataTreeView import DataTreeView
from ..custom.buttonDesigns import BigArrowButton, BigPlusButton, SubsetDataButton, ViewDataButton
from ..custom.tableviews.ICDataTable import PandaTableDialog
from ..custom.ICLiveGraph import LiveGraph
from ..custom.analysisSelection import AnalysisSelection
from ..utils import removeFileExtension, areFilesSuitableToLoad, createLabel
from ..custom.warnMessage import WarningMessage
import os
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

class LoadButton(BigArrowButton):
    ""
    def __init__(self,parent=None, callback = None, *args,**kwargs):
        super(LoadButton,self).__init__(parent=parent,*args,**kwargs)
        self.callback = callback
        self.setAcceptDrops(True)
        self.acceptDrop = False
    
    def dragMoveEvent(self, e):
        "Ignore/acccept drag Move Event"
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()
    
    def dragEnterEvent(self,event):
        "check if drag event has urls (e.g. files)"
        #check if drop event has Urls
        if event.mimeData().hasUrls():
            event.accept()
            self.acceptDrop = True
        else:
            event.ignore()
            self.acceptDrop = False
    
    def dropEvent(self,event):
        "Allows drop of files from os"
        #find Urls of dragged objects
        droppedFiles = [url.path() for url in event.mimeData().urls() if url.isValid()]
        #check if file ends with proper fileExtension
        checkedDroppedFiles = areFilesSuitableToLoad(droppedFiles)
        if len(checkedDroppedFiles) > 0:
            event.accept()
            self.callback(checkedDroppedFiles)


class DropButton(QPushButton):
    ""
    def __init__(self,parent=None, callback = None, acceptedDragTypes = ["Categories"], toolTipStr = None, *args,**kwargs):
        super(DropButton,self).__init__(parent=parent,*args,**kwargs)
        self.callback = callback
        self.acceptDrop = False
        self.acceptedDragTypes = acceptedDragTypes
        self.setAcceptDrops(True)
        if toolTipStr is not None:
            self.setToolTip(toolTipStr)

    def dragEnterEvent(self,event):
        "Check if drag items is of correct datatype"
        #check if drop event has Urls
       
        dragType = self.parent().parent().parent().parent().getDragType()
        #check if type is accpeted and check if not all items are anyway already there
        if dragType in self.acceptedDragTypes:
            self.acceptDrop = True
        else:
            self.acceptDrop = False
        
        event.accept()
    
    def dragMoveEvent(self, e):
        "Ignore/acccept drag Move Event"
        if self.acceptDrop:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self,event):
        "Initiate callback"
        #find Urls of dragged objects
        try:
            event.accept()
            self.callback()
           # print("calling")
        except Exception as e:
            print(e)
        


class DataHandleFrame(QFrame):

    def __init__(self,parent=None, mainController = None):
        super(DataHandleFrame, self).__init__(parent)

        self.mC = mainController
        self.__controls()
        self.__layout() 
        self.addShortcuts()

        self.setLineWidth(2)
        self.setMidLineWidth(2)
        

       # self.setStyleSheet("""QFrame {background-color: #E8E8E8;margin:1px; border:1px solid black}""")

    def __controls(self):
        self.bigFrame = QFrame(self)
        self.bigFrame.setLayout(QVBoxLayout())
        self.bigFrame.layout().setContentsMargins(0,0,0,0)

        self.frames = CollapsableFrames(parent=self.bigFrame,buttonDesign=CollapsButton)
        self.qS = QuickSelect(parent=self,sendToThreadFn = self.sendToThread, mainController=self.mC) 
        self.liveGraph = LiveGraph(self,self.mC) ## could als be retrieved from parent?
        self.analysisSelection = AnalysisSelection(self,mainController=self.mC)
        self.bigFrame.layout().addWidget(self.frames)
        
        vbox1 = QHBoxLayout()
        loadDataButton = BigPlusButton(callback = self.addTxtFiles, tooltipStr ="Load Data from file.\nThis will reset the view.")
        
        #addDataButton = BigPlusButton()
        #addDataButton.clicked.connect(self.loadSession)

        loadSessionButton = BigArrowButton(direction="up", tooltipStr="Load session.")
        loadSessionButton.clicked.connect(self.loadSession)

        saveSessionButton = BigArrowButton(direction="down", tooltipStr="Saves session. Note: the current figure is not saved.")
        saveSessionButton.clicked.connect(self.saveSession)

        viewDataButton = ViewDataButton(self, tooltipStr="View selected data.")
        viewDataButton.clicked.connect(self.showData)
        subsetDataButton = SubsetDataButton(#DropButton(parent=self,
            callback = self.subsetData,
            getDragType= self.getDragType,
            acceptDrops= True,
            tooltipStr = "Drop categorical columns to split data on unique values.\nNaN Object String ('-') will be ignored.")
        
        vbox1.addWidget(loadDataButton)
        #vbox1.addWidget(addDataButton)
        vbox1.addStretch(1)
        vbox1.addWidget(loadSessionButton)
        vbox1.addWidget(saveSessionButton)
        vbox1.addStretch(3)
        vbox1.addWidget(viewDataButton)
        vbox1.addWidget(subsetDataButton)
        #vbox1.addStretch(1)
        loadDataButton.clicked.connect(self.askForFile)
        vbox2 = QVBoxLayout()
        self.dataTreeView = CollapsableDataTreeView(self, sendToThreadFn = self.sendToThread, mainController = self.mC)
        vbox2.addWidget(self.dataTreeView)
        vbox2.setContentsMargins(0,0,0,0)
        
        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.qS)
        vbox3.setContentsMargins(0,0,0,0)
        vbox4 = QVBoxLayout()
        vbox4.addWidget(self.liveGraph)

        vbox5= QVBoxLayout()
        vbox5.addWidget(self.analysisSelection)
        vbox5.setContentsMargins(3,3,3,3)

        
        frameWidgets = [
                {"title":"Load Data & Sessions","open":True,"fixedHeight":True,"height":50,"layout":vbox1},
                {"title":"Data","open":True,"fixedHeight":False,"height":0.4,"layout":vbox2},
                {"title":"Quick Select","open":False,"fixedHeight":False,"height":0.4,"layout":vbox3},
                {"title":"Live Graph","open":False,"fixedHeight":False,"height":0.4,"layout":vbox4},
                {"title":"Analysis","open":False,"fixedHeight":False,"height":150,"layout":vbox5}]
        self.frames.addCollapsableFrame(frameWidgets,widgetHeight=20,fontSize=9)
        
    def addShortcuts(self):
        "Add Shortcuts for copying/pasting data."
        self.ctrlV = QShortcut(QKeySequence("Ctrl+v"), self)
        self.ctrlV.activated.connect(self.readClipboard)

        self.ctrlC = QShortcut(QKeySequence("Ctrl+c"), self)
        self.ctrlC.activated.connect(self.copyToClipboard)

    def askForFile(self):
        "Get File Names"
        dlg = QFileDialog(caption="Select File",filter = "ICLoad Files (*.txt *.csv *tsv *xlsx);;Text files (*.txt *.csv *tsv);;Excel files (*.xlsx)")
        dlg.setFileMode(QFileDialog.ExistingFiles)

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            self.openDialog(filenames)

    def copyToClipboard(self,event=None):
        "Send copy data request to thread"
        funcProps = {"key":"data::copyDataFrameSelection","kwargs":{}}
        self.dataTreeView.sendToThread(funcProps,addSelectionOfAllDataTypes=True,addDataID=True)

    def getSelectedColumns(self,dataType="all"):
        "Get Selected columns"
        return self.dataTreeView.getSelectedColumns(dataType)

    def getColumns(self,dataType = "all"):
        ""
        return self.dataTreeView.getColumns(dataType=dataType)

    def getDragColumns(self):

        return self.dataTreeView.getDragColumns()
    
    def getTreeView(self,dataHeader = "Numeric Floats"):
        ""
        return self.dataTreeView.getTreeView(dataHeader)

    def getDragType(self):

        return self.dataTreeView.getDragType() 

    def getDataID(self):
        ""
        return self.dataTreeView.getDataID()

    def openDialog(self, selectedFiles = []):
        ""
        try:
            if any(f.endswith(".txt") for f in selectedFiles):
                d = PlainTextImporter(mainController = self.mC) 
                d.exec_()
                if d.result():
                    #set replace object
                    txtFiles = [f for f in selectedFiles if f.endswith(".txt")]
                    replaceObjectNan = d.replaceObjectNan
                    self.mC.config.setParam("replaceObjectNan",replaceObjectNan)
                    #load files on thread
                    self.addTxtFiles(txtFiles,d.getSettings())

            if any(f.endswith(".xlsx") for f in selectedFiles):
                d = ExcelImporter(mainController = self.mC) 
                d.exec_()
                if d.result():
                    #set replace object
                    xlsxFiles = [f for f in selectedFiles if f.endswith(".xlsx")]
                    #load files on thread
                    self.addExcelFiles(xlsxFiles,d.getSettings())
        except Exception as e:
            print(e)

    def addExcelFiles(self,files, loadFileProps = None):
        ""
        files = [f for f in files if f.endswith(".xlsx")]
        for filePath in files:
            if os.path.exists(filePath):
                fileName = removeFileExtension(Path(filePath).name)
                self.mC.config.setParam("WorkingDirectory",os.path.dirname(filePath))
                funcProps = dict()
                funcProps["key"] = "data::addDataFrameFromExcelFile"
                funcProps["kwargs"] = {"fileName":fileName,
                                    "pathToFile":filePath,
                                    "loadFileProps":loadFileProps}
                #self.mC.ICDataManger.loadDf.emit(filePath,fileName,loadFileProps)
                
                self.mC.sendRequestToThread(funcProps)


    def addTxtFiles(self,files, loadFileProps = None):
        ""
        files = [f for f in files if f.endswith(".txt")]
        for filePath in files:
            if os.path.exists(filePath):
                fileName = removeFileExtension(Path(filePath).name)
                self.mC.config.setParam("WorkingDirectory",os.path.dirname(filePath))
                funcProps = dict()
                funcProps["key"] = "data::addDataFrameFromTxtFile"
                funcProps["kwargs"] = {"fileName":fileName,
                                    "pathToFile":filePath,
                                    "loadFileProps":loadFileProps}
                #self.mC.ICDataManger.loadDf.emit(filePath,fileName,loadFileProps)
                
                self.mC.sendRequestToThread(funcProps)

    def subsetData(self):
        ""
        columnNames = self.getDragColumns()
        dataID = self.getDataID()
        funcProps = {"key":"filter::splitDataFrame","kwargs":{"dataID" : dataID,"columnNames":columnNames}}
        self.mC.sendRequestToThread(funcProps)

    def deleteData(self):
        ""
        if not self.mC.data.hasData():
            return
        dataID = self.getDataID()
        funcProps = {"key":"data::deleteData","kwargs":{"dataID" : dataID}}
        self.mC.sendRequestToThread(funcProps)

    def exportData(self, exportDataFormat = "txt"):
        ""
        if not self.mC.data.hasData():
            return
        dataID = self.getDataID()

        if dataID is not None:
            baseFileName = self.mC.data.getFileNameByID(dataID)
            
            if exportDataFormat == "txt":
                fileDataFormat = "Text/CSV files (*.txt *.csv)"
            elif exportDataFormat == "xlsx":
                fileDataFormat = "Excel files (*.xlsx)"
            elif exportDataFormat == "json":                            
                fileDataFormat = "Json files (*.json)"
            elif exportDataFormat == "md":
                fileDataFormat = "Markdown files (*.md)"
                #if user cancels file selection, return function
            else:
                return
            if not baseFileName.endswith(fileDataFormat):
                baseFileName = "{}.{}".format(baseFileName,exportDataFormat)

            baseFilePath = os.path.join(self.mC.config.getParam("WorkingDirectory"),baseFileName)
            fname,_ = QFileDialog.getSaveFileName(self, 'Save file', baseFilePath,fileDataFormat)

            if fname:
                columnOrder = self.getColumns()
                funcProps = {"key":"data::exportData","kwargs":{"dataID" : dataID,"path":fname, "columnOrder":columnOrder, "fileFormat" : exportDataFormat}}
                self.mC.sendRequestToThread(funcProps)
    
    def showData(self):
        ""
       
        if not self.mC.data.hasData():
            warn  = WarningMessage(infoText="No data found. Please load data first.",iconDir = self.mC.mainPath)
            warn.exec_()
            return
        try:
            dataID = self.getDataID()
            columnNames = self.mC.data.getPlainColumnNames(dataID)
            useClipping = self.mC.config.getParam("data.view.ignore.clipping")
            dataFrame = self.mC.data.getDataByColumnNames(dataID,columnNames, ignore_clipping = not useClipping)["fnKwargs"]["data"]
            dlg = PandaTableDialog(mainController = self.mC ,df = dataFrame, parent=self)
            dlg.exec_()
        except Exception as e:
            print(e)

    def updateFilter(self,boolIndicator,resetData=False):

        self.dlg.updateModelDataByBool(boolIndicator,resetData)

    def updateDataInTreeView(self,columnNamesByType, dataID = None):
        "Updating data in treeview (Numeric Floats, Integers, Categories)"
        if dataID is not None:
            if not self.dataTreeView.getDataID() == dataID:
                return
        self.dataTreeView.updateDataInTreeView(columnNamesByType)
        #update grouping
        self.dataTreeView.dataHeaders["Numeric Floats"].setCurrentGrouping()

    def updateDataFrames(self,dfs,selectLastDf=True):
        ""
        if isinstance(dfs,dict):
            self.dataTreeView.updateDfs(dfs,selectLastDf)

    def updateDataInQuickSelect(self,data):
        """ """
        self.qS.addData(data)

    def updateColorAndSizeInQuickSelect(self,checkedColors=None,checkedSizes=None):
        ""
        
        self.qS.updateColorsAndSizes(checkedColors,checkedSizes)
      
      
    def sendToThread(self,funcProps):
        
        try:
            self.mC.sendRequestToThread(funcProps)
        except Exception as e:
            print(e)   

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.bigFrame)
        self.layout().setContentsMargins(0,0,0,0)


    def readClipboard(self,event=None):
        ""
        funcKey = "data::addDataFrameFromClipboard"
        self.mC.sendRequestToThread(funcProps = {"key":funcKey,"kwargs":{}})


    def saveSession(self):
        ""
        workingDir = self.mC.config.getParam("WorkingDirectory")
        currentTime = datetime.date.today()
        fileName,_ = QFileDialog.getSaveFileName(self,"Save Instant Clue Session",os.path.join(workingDir,"{}.ic".format(currentTime)),"Instant Clue File (*.ic)")
        if fileName is not None and fileName != "":
            
            self.mC.sessionManager.saveSession(fileName)
            self.mC.sendMessageRequest({"title":"Saved ..","message":"Session saved."})

    def loadSession(self):
        ""
        workingDir = self.mC.config.getParam("WorkingDirectory")
        fileName,_ = QFileDialog.getOpenFileName(self,"Load Instant Clue Session",workingDir,"Instant Clue File (*.ic)")
        if fileName is not None and fileName != "":
            self.mC.sendRequestToThread(funcProps = {"key":"session::load","kwargs":{"sessionPath":fileName}})
            self.mC.sendMessageRequest({"title":"Saved ..","message":"Session loaded."})