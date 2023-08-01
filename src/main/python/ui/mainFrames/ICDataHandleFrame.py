
from fileinput import filename
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..dialogs.DataFrames.ICLoadFileDialogs import PlainTextImporter, ExcelImporter
from ..dialogs.Connect.ICFetchDataFromMitoCube import ICFetchDataFromMitoCube
from ..custom.ICCollapsableFrames import CollapsableFrames
from ..custom.Widgets.ICButtonDesgins import CollapsButton
from ..custom.ICQuickSelect import QuickSelect
from ..custom.dataFrameSelection import CollapsableDataTreeView
from ..custom.Widgets.ICButtonDesgins import BigArrowButton, BigPlusButton, ViewDataButton
from ..custom.tableviews.ICDataTable import PandaTableDialog
from ..custom.ICLiveGraph import LiveGraph
from ..custom.analysisSelection import AnalysisSelection
from ..utils import removeFileExtension, areFilesSuitableToLoad, getHoverColor, getCollapsableButtonBG
from ..custom.warnMessage import WarningMessage


import os
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import socket


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
        super(DataHandleFrame, self).__init__(parent=parent)

        self.mC = mainController
        self.__controls()
        self.__layout() 
        self.addShortcuts()

        self.setLineWidth(2)
        self.setMidLineWidth(2)
        

    def __controls(self):
        self.bigFrame = QFrame(self)
        self.bigFrame.setLayout(QVBoxLayout())
        self.bigFrame.layout().setContentsMargins(0,0,0,0)

        self.frames = CollapsableFrames(parent=self.bigFrame,buttonDesign=CollapsButton)
        self.qS = QuickSelect(parent=self,mainController=self.mC) 
        self.liveGraph = LiveGraph(self,self.mC) ## could als be retrieved from parent?
        self.analysisSelection = AnalysisSelection(self,mainController=self.mC)
        self.bigFrame.layout().addWidget(self.frames)
        
        vbox1 = QHBoxLayout()
        loadDataButton = BigPlusButton(
                parent = self,
                callback = self.addTxtFiles, 
                tooltipStr ="Load Data from file.\nThis will reset the current plot.",
                menuFn = self.showSettingMenu,
                menuKeyWord = "Load Data")
        
        #addDataButton = BigPlusButton()
        #addDataButton.clicked.connect(self.loadSession)

        loadSessionButton = BigArrowButton(parent = self,direction="up", tooltipStr="Load session.")
        loadSessionButton.clicked.connect(self.loadSession)

        saveSessionButton = BigArrowButton(parent = self,direction="down", tooltipStr="Saves session. Note: the current figure is not saved with full properties\nMake sure to save the figure before.")
        saveSessionButton.clicked.connect(self.saveSession)

        viewDataButton = ViewDataButton(self, 
                    tooltipStr="View selected data. The table allow you to filter and sort data.",
                    menuFn = self.showSettingMenu,
                    menuKeyWord = "Data View")
        viewDataButton.clicked.connect(self.showData)


        
        vbox1.addWidget(loadDataButton)
        #vbox1.addWidget(addDataButton)
        vbox1.addStretch(1)
        vbox1.addWidget(loadSessionButton)
        vbox1.addWidget(saveSessionButton)
        vbox1.addStretch(3)
        vbox1.addWidget(viewDataButton)
       # vbox1.addWidget(subsetDataButton)
        #vbox1.addStretch(1)
        loadDataButton.clicked.connect(self.askForFile)
        vbox2 = QVBoxLayout()
        self.dataTreeView = CollapsableDataTreeView(parent=self, mainController = self.mC)
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
                {"title":"Load & Save Data","open":True,"fixedHeight":True,"height":50,"layout":vbox1},
                {"title":"Data","open":True,"fixedHeight":False,"height":0.4,"layout":vbox2},
                {"title":"Quick Select","open":False,"fixedHeight":False,"height":0.4,"layout":vbox3},
                {"title":"Live Graph","open":False,"fixedHeight":False,"height":0.4,"layout":vbox4},
                {"title":"Analysis","open":False,"fixedHeight":False,"height":150,"layout":vbox5}]
        self.frames.addCollapsableFrame(frameWidgets,widgetHeight=20,fontSize=9,
            closeColor = getCollapsableButtonBG(),
            openColor = getCollapsableButtonBG(),
            hoverColor = getHoverColor())
        
    def addShortcuts(self):
        "Add Shortcuts for copying/pasting data."
        self.ctrlV = QShortcut(QKeySequence("Ctrl+v"), self)
        self.ctrlV.activated.connect(self.readClipboard)

        
    
        self.ctrlC = QShortcut(QKeySequence("Ctrl+c"), self)
        self.ctrlC.activated.connect(self.copyToClipboard)

        self.ctrlH = QShortcut(QKeySequence("Ctrl+f"), self)
        self.ctrlH.activated.connect(self.openFindReplaceDialog)


    

    def askForFile(self):
        "Get File Names"
        dlg = QFileDialog(caption="Select File",filter = "ICLoad Files (*.txt *.csv *tsv *xlsx);;Text files (*.txt *.csv *tsv);;Excel files (*.xlsx)")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)

        if dlg.exec():
            filenames = dlg.selectedFiles()
            self.openDialog(filenames)

    def copyToClipboard(self,event=None):
        "Send copy data request to thread"
       
        funcProps = {"key":"data::copyDataFrameSelection","kwargs":{}}
        self.dataTreeView.sendToThread(funcProps,addSelectionOfAllDataTypes=True,addDataID=True)

    def copyDataFrameToClipboard(self,e=None):
        "Copy data frame to clipbaord. Active clipping will NOT be ignored."
        funcProps = {"key":"data:copyDataToClipboard","kwargs":{"dataID":self.mC.getDataID()}} 
        self.mC.sendRequestToThread(funcProps)

    def fetchDataFromMitoCube(self):
        ""
        dlg = ICFetchDataFromMitoCube(mainController=self.mC)
        dlg.exec()

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
            if any(f.endswith(".txt") | f.endswith(".csv") | f.endswith(".tsv") for f in selectedFiles):
                d = PlainTextImporter(mainController = self.mC) 
                d.exec()
                if d.result():
                    #set replace object
                    txtFiles = [f for f in selectedFiles if f.endswith(".txt")]
                    replaceObjectNan = d.replaceObjectNan
                    self.mC.config.setParam("replaceObjectNan",replaceObjectNan)
                    #load files on thread
                    self.addTxtFiles(txtFiles,d.getSettings())

            if any(f.endswith(".xlsx") for f in selectedFiles):
               
                self.excelImporter = ExcelImporter(mainController = self.mC, selectedFiles = selectedFiles) 
                
                self.excelImporter.exec()
                
        except Exception as e:
            print(e)


    def updateExcelFileInDialog(self,fileSheetNames):
        ""
        if hasattr(self,"excelImporter"):
            self.excelImporter.setIsLoading(False)
            #self.excelImporter.setExcelFiles(excelFiles)
            self.excelImporter.setSheetsToSelectTable(fileSheetNames["df"])

    def openFindReplaceDialog(self,e=None):
        ""
        if self.mC.data.hasData():
            self.dataTreeView.findAndReplace()

    def addExcelFiles(self,files):
        ""
        files = [f for f in files if f.endswith(".xlsx")]
        self.openDialog(selectedFiles=files)
        # 


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

    def deleteData(self):
        ""
        if not self.mC.data.hasData():
            return
        dataID = self.getDataID()
        funcProps = {"key":"data::deleteData","kwargs":{"dataID" : dataID}}
        self.mC.sendRequest(funcProps) #no thread needed super fast..


    def exportMultipleDataFramesToExcel(self,exportDataFormat):
        ""
        if len(self.mC.data.dfs) > 1 and exportDataFormat == "xlsx-multiple":
    
            selectedItemsIdx = self.mC.askForItemSelection(items = pd.Series(self.mC.data.getFileNames())).index.values.tolist()
            if selectedItemsIdx is None: return 
    
            dataIDs = self.mC.data.getDataIDbyFileNameIndex(idx=selectedItemsIdx)
        
            selectedItems = [self.mC.data.getFileNames()[n] for n in selectedItemsIdx]
        
        else:
            dataIDs = [self.getDataID()]
            selectedItems = [self.mC.data.fileNameByID[dataIDs[0]]]

        data = [self.mC.getDataByDataID(dataID,useClipping=True) for dataID in dataIDs]
        softwareParams = [("Software","Instant Clue"),
					("Version",self.mC.version),
                    ("Computer Name",socket.gethostname()),
                    ("Date",datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")),
                    ("FileNames",", ".join(selectedItems))
                    ]

        fileName = self.mC.askForExcelFileName(defaultName = "{}.xlsx".format(selectedItems[0]) if len(selectedItems) == 1 else "MultiExcelExport.xlsx")
        if fileName != "":
            fkey = "data::exportDataToExcel"
            kwargs = {
                "pathToExcel": fileName,
                "fileNames": selectedItems,
                "dataFrames": data,
                "softwareParams" : softwareParams,
                "groupings" : self.mC.grouping.getGroupings()
            }
            funcProps = dict()
            funcProps["key"] = fkey
            funcProps["kwargs"] = kwargs

            self.mC.sendRequestToThread(funcProps)

    def exportData(self, exportDataFormat = "txt"):
        ""
        if not self.mC.data.hasData():
            return

        if exportDataFormat.startswith("xlsx"):
            self.exportMultipleDataFramesToExcel(exportDataFormat)
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
            warn.exec()
            return
       
        dataID = self.getDataID()
        useClipping = self.mC.config.getParam("data.view.use.clipping")
        clippingActive = self.mC.data.hasClipping(dataID)
        dataFrame = self.mC.getDataByDataID(dataID,useClipping=useClipping)
        self.openDataFrameinDialog(dataFrame,clippingActive = clippingActive)

    def openDataFrameinDialog(self,dataFrame,*args,**kwargs):
        ""
        dlg = PandaTableDialog(mainController = self.mC ,df = dataFrame, parent=self,*args,**kwargs)
        dlg.exec()
        

    def updateFilter(self,boolIndicator,resetData=False):
        ""
        self.dlg.updateModelDataByBool(boolIndicator,resetData)

    def updateDataInTreeView(self,columnNamesByType, tooltipData = {}, dataID = None):
        "Updating data in treeview (Numeric Floats, Integers, Categories)"
        if dataID is not None:
            if not self.dataTreeView.getDataID() == dataID:
                return
       # self.dataTreeView.updateData.emit(columnNamesByType, tooltipData, dataID)
        self.dataTreeView.updateDataInTreeView(columnNamesByType, tooltipData)
        self.updateGroupingInTreeView()
    
    def updateGroupingInTreeView(self):
        #update grouping
        
        self.dataTreeView.dataHeaders["Numeric Floats"].setCurrentGrouping()

    def updateDataFrames(self,dfs,selectLastDf=True,dataComboboxIndex=None,sessionIsBeeingLoaded=False, remainLastSelection = False):
        ""
        if isinstance(dfs,dict):
            
            self.dataTreeView.updateDfs(dfs,selectLastDf,
                    specificIndex=dataComboboxIndex,
                    sessionIsBeeingLoaded = sessionIsBeeingLoaded, 
                    remainLastSelection=remainLastSelection)

    def updateDataInQuickSelect(self,data):
        ""
        self.qS.dataChanged.emit(data)

    def updateColorAndSizeInQuickSelect(self,checkedColors=None,checkedSizes=None):
        ""
        self.qS.updateColorsAndSizes(checkedColors,checkedSizes)
      
    def sendToThread(self,funcProps):
        ""
        self.mC.sendRequestToThread(funcProps)
       

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
            
    def showSettingMenu(self,menuKeyWord,sender):
        ""
        bottomLeft = self.findSendersBottomLeft(sender)
        menu = self.mC.createSettingMenu(menuKeyWord)
        menu.exec(bottomLeft)
        
    def findSendersBottomLeft(self,sender):
        ""
        #find bottom left corner
        senderGeom = sender.geometry()
        bottomLeft = sender.parent().mapToGlobal(senderGeom.bottomLeft())
        #set sender status 
        if hasattr(sender,"mouseOver"):
            sender.mouseOver = False
        return bottomLeft
