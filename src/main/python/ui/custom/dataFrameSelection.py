from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

#internal imports
from .ICCollapsableFrames import CollapsableFrames
from .Widgets.ICButtonDesgins import DataHeaderButton, ViewHideIcon, FindReplaceButton, ResetButton, BigArrowButton, ICStandardButton
from .ICDataTreeView import DataTreeView
from ..dialogs.ICDFindReplace import FindReplaceDialog
from ..dialogs.Transformations.ICMultiBlockSGCCA import ICMultiBlockSGCCA
from ..dialogs.DataFrames.ICDMergeDataFrames import ICDMergeDataFrames
from ..dialogs.DataFrames.ICCorrelateDataFrames import ICCorrelateDataFrames, ICCorrelateFeatures
from ..dialogs.OmicsTools.ICSampleList import ICSampleListCreater
from ..dialogs.OmicsTools.ICProteinPeptideView import ICProteinProteinView
from .utils import dataFileExport
from ..custom.warnMessage import AskForFile, WarningMessage, AskStringMessage
from ..utils import WIDGET_HOVER_COLOR, getHoverColor, INSTANT_CLUE_BLUE, getStandardFont, createMenu, createSubMenu, getCollapsableButtonBG


#external imports
import pandas as pd
from collections import OrderedDict
import os 
import time
import datetime

dataTypeMenu = ["Sort columns .."]

menuFuncs = [
    {
        "subM":"Sort columns ..",
        "name":"Alphabetically",
        "funcKey": "sortLabels",
    },
    {
        "subM":"Sort columns ..",
        "name":"Custom order",
        "funcKey": "customSortLabels",
    }
]


class CollapsableDataTreeView(QWidget):
    
    def __init__(self, parent=None,dfs = OrderedDict(), mainController = None):
        super(CollapsableDataTreeView, self).__init__(parent)
    
        self.mC = mainController
        self.dfs = dfs #OrderedDict keys: dataID, values: names
        self.preventReset = False
        self.sessionIsBeeingLoaded = False
        self.__controls()
        self.__layout() 
        self.__connectEvents()
        self.updateDfs()
        self.dataID = None

    def __controls(self):
        #set up data frame combobox and its style
        self.combo = QComboBox(self)
        self.combo.setStyleSheet("selection-background-color: white; outline: None; selection-color: {}".format(INSTANT_CLUE_BLUE)) 
        ### add menu button
        self.menuButton = ICStandardButton(itemName = "...", tooltipStr="Menu for mutliple settings such as grouping.")
        self.menuButton.setFixedSize(15,15)                     
        #find & replace button
        self.findReplaceButton = FindReplaceButton()
        self.findReplaceButton.setToolTip("Find & replace in column headers as well as in the data frame.")
        #add function
        self.findReplaceButton.clicked.connect(self.findAndReplace)
        #set up hide shortcuts
        self.hideSC = ViewHideIcon(self)
        self.hideSC.clicked.connect(self.hideShortCuts)
        #export 
        self.exportButton = BigArrowButton(self,tooltipStr="Export selected data to txt file. Right-click to see more options such Excel, Json, Markdown.", buttonSize=(15,15))
        
        #set up delete button
        self.deleteButton = ResetButton(tooltipStr="Delete selected data.")
        self.deleteButton.clicked.connect(self.deleteData)
        # set up collapsable frame widget
        # we need extra frame to get parent size correctly
        self.dataTreeFrame = QFrame(self)
        
        
        #add widget to frame 
        self.frames = CollapsableFrames(parent = self, buttonDesign = DataHeaderButton, spacing = 0, buttonMenu =  self.reportMenuRequest)
        frameWidgets = []
        self.dataHeaders = dict()
        for header in ["Numeric Floats","Integers","Categories"]:
            self.dataHeaders[header] = DataTreeView(self, tableID = header, mainController=self.mC)
            frame = {"title":header,
                     "open":False,
                     "fixedHeight":False,
                     "height":0,
                     "layout":self.dataHeaders[header].layout()}
            frameWidgets.append(frame)
            
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
        hbox1 = QHBoxLayout() 
        hbox1.addWidget(self.combo)
        hbox1.addWidget(self.menuButton)
        hbox1.addWidget(self.findReplaceButton)
        hbox1.addWidget(self.hideSC)
        hbox1.addWidget(self.exportButton)
        hbox1.addWidget(self.deleteButton)

        self.layout().addLayout(hbox1)
        self.layout().addWidget(self.dataTreeFrame)

        self.dataTreeFrame.setLayout(QVBoxLayout())
        self.dataTreeFrame.layout().setContentsMargins(0,0,0,0)
        self.dataTreeFrame.layout().addWidget(self.frames)

        self.layout().setContentsMargins(0,0,0,0)
        self.layout().setSpacing(1)

    def __connectEvents(self):
        ""
        self.combo.currentIndexChanged.connect(self.dfSelectionChanged)
        self.menuButton.clicked.connect(self.showMenu)
        self.exportButton.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.exportButton.clicked.connect(self.exportData)
        self.exportButton.customContextMenuRequested.connect(self.exportMenu)

    def addDataFrame(self, dataID, dataFrameName):
        "Add a new data frame to the combobox"
        if dataID not in self.dfs:
            self.dfs[dataID] = dataFrameName
            self.updateDfs()
            self.dfSelectionChanged(len(self.dfs)-1)
        
    def reportMenuRequest(self,dataType, menuPosition):
        ""
        if self.dataID is not None:
            sender = self.sender()
            if hasattr(sender,"loseFocus"):
                sender.loseFocus()
            menus = createSubMenu(subMenus=dataTypeMenu)

            for menuItem in menuFuncs:
                action = menus[menuItem["subM"]].addAction(menuItem["name"])
                if dataType not in self.dataHeaders:
                    dataType = dataType[:-5].strip()
                if dataType in self.dataHeaders:
                    action.triggered.connect(getattr(self.dataHeaders[dataType],menuItem["funcKey"]))

            menus["main"].exec(menuPosition)

    def addSelectionOfAllDataTypes(self,funcProps):
        """
            This functions helps to add column selections from all dataTreeViews 
            (numeric,int,categories). Especially used when sending Requests to Thread.
        """
        if isinstance(funcProps,dict):
            if "kwargs" in funcProps:
                if "columnNames" in funcProps and isinstance(funcProps["kwargs"]["columnNames"],list):
                    funcProps["kwargs"]["columnNames"] = funcProps["kwargs"]["columnNames"] + self.getSelectedColumns()
                else:
                    funcProps["kwargs"]["columnNames"] = self.getSelectedColumns()

        return funcProps

    def dfSelectionChanged(self, comboIndex):
        ""
        dataID = self.getDfId(comboIndex)
        if dataID is not None and self.dataID != dataID:
            #send back to main
            
            funcProps = {
                "key":"data::getColumnNamesByDataID" if not self.sessionIsBeeingLoaded else "data::getColumnNamesByDataIDSilently",
                "kwargs":{"dataID":dataID}}
            self.dataID = dataID
            self.mC.mainFrames["data"].qS.resetView(updatePlot=False)
            self.mC.mainFrames["data"].liveGraph.clearGraph()
            self.updateDataIDInTreeViews()
            self.mC.sendRequestToThread(funcProps)
            self.sessionIsBeeingLoaded = False
            #print(dataID,"updated",datetime.datetime.now(), funcProps)
           

    def deleteData(self,e=None):
        ""
        self.mC.mainFrames["data"].deleteData()

    def exportData(self,e=None):
        ""
        self.mC.mainFrames["data"].exportData()
        
    def exportMenu(self,e=None):
        ""
        sender = self.sender()
        if hasattr(sender,"mouseLostFocus"):
            sender.mouseLostFocus()
        menu = createMenu()
        
        for fileFormat, actionName in dataFileExport:

            action = menu.addAction(actionName)
            action.triggered.connect(lambda _, txtFileFormat = fileFormat: self.mC.mainFrames["data"].exportData(txtFileFormat))
        
        action = menu.addAction("clipboard")
        action.triggered.connect(self.mC.mainFrames["data"].copyDataFrameToClipboard)

        senderGeom = self.sender().geometry()
        bottomLeft = self.mapToGlobal(senderGeom.bottomLeft())
        menu.exec(bottomLeft) 

    def getDfId(self,comboIndex):
        "Return DataID from index selection"
        dfIds = list(self.dfs.keys()) #dfs must be a ordered dict
        if comboIndex < len(dfIds) and comboIndex >= 0:
            return dfIds[comboIndex]

    def getSelectedColumns(self, dataType = "all"):
        ""
        selectedColumns = []
        for dataHeader, treeView in self.dataHeaders.items():
            if dataType == "all" or dataType == dataHeader:
                selectedColumns.extend(treeView.getSelectedData().values.tolist())
        return selectedColumns

    def getColumns(self, dataType):
        currentColumns = OrderedDict()
        for dataHeader, treeView in self.dataHeaders.items():
            if dataType == "all" or dataHeader == dataType:
                currentColumns[dataHeader] = treeView.getData()
        return currentColumns


    def getDragColumns(self):
        ""
        if hasattr(self,"draggedColumns"):
            return self.draggedColumns
        
    def getDragType(self):
        if hasattr(self,"dragType"):
            return self.dragType
    
    def getDataID(self):
        ""
        if hasattr(self,"dataID"):
            return self.dataID 

    def getTreeView(self,dataHeader = "Numeric Floats"):
        ""
        if dataHeader in self.dataHeaders:
            return self.dataHeaders[dataHeader]

    def updateDragData(self, draggedColumns, dragType):
        """
        Dragged Columns and dragType is stored and can be accesed by 
        function getDragColumns and getDragType
        """
        self.draggedColumns = draggedColumns
        self.dragType = dragType

    def findAndReplace(self):
        ""
        try:
            if self.mC.data.hasData():

                senderGeom = self.findReplaceButton.geometry()
                bottomLeft = self.parent().mapToGlobal(senderGeom.bottomLeft())

                frd = FindReplaceDialog(self.mC)
                frd.setGeometry(bottomLeft.x(),bottomLeft.y(),200,150)
                if frd.exec():
                    funcKey = "data::replace"
                    fS = frd.findStrings 
                    rS = frd.replaceStrings
                    specificColumnSelected = frd.specificColumnSelected #bool
                    selectedIndex = frd.selectedColumnIndex
                    dataType = frd.selectedDataType # selected data type
                    mustMatchCompleteCell = frd.mustMatchCompleteCell 
                    if selectedIndex == 0: #this means complete selection, save if colum header is by change same in data

                        specificColumn = self.getSelectedColumns(dataType=dataType)
                        if len(specificColumn) == 0:
                            self.mC.sendToWarningDialog(infoText = "No selected columns found in selected data type: {}".format(dataType))
                            return
                        
                    #  specificColumn = frd.selectedColumn
                    elif specificColumnSelected:
                        specificColumn = [frd.selectedColumn]
                    else:
                        specificColumn = None
                    
                    funcProps = {"key":funcKey,"kwargs":{
                                        "findStrings":fS,
                                        "replaceStrings":rS,
                                        "specificColumns":specificColumn,
                                        "dataID":self.mC.getDataID(),
                                        "dataType":dataType,
                                        "mustMatchCompleteCell":mustMatchCompleteCell}
                                }
                    self.mC.sendRequestToThread(funcProps)
            else:
                self.mC.sendToInformationDialog(infoText="Please load data first.")
        except Exception as e:
            print(e)

    def hideShortCuts(self):
        "User can hide/show shortcuts"
        self.sender().stateChanged()
        for treeView in self.dataHeaders.values():
            treeView.hideShowShortCuts()

    def updateDataInTreeView(self,columnNamesByType, tooltipData = {}):
        """Add data to the data treeview"""
        print("updating data -- debug")
        if isinstance(columnNamesByType,dict):
            for headerName, values in columnNamesByType.items():
                if headerName in self.dataHeaders:
                    if isinstance(values,pd.Series):
                        self.dataHeaders[headerName].updateData.emit(values, tooltipData) #addData(values, tooltipData) 
                        self.frames.headerNameChanged.emit(headerName,"{} ({})".format(headerName,values.size))
                        #self.frames.setHeaderNameByFrameID(headerName,"{} ({})".format(headerName,values.size))
                        if values.size == 0:
                            self.frames.headerStateChange.emit(headerName)
                    else:
                        raise ValueError("Provided Data are not a pandas Series!") 

    def updateDataIDInTreeViews(self):
        "Update Data in Treeview:: settingData"
        for treeView in self.dataHeaders.values():
            treeView.setDataID(self.dataID)

    def updateDfs(self, dfs = None, selectLastDf = True, remainLastSelection = False, specificIndex = None, sessionIsBeeingLoaded = False):
        "Updates the data frame selection."
        if dfs is not None:
            self.dfs = dfs
            if remainLastSelection:
                lastIndex = self.combo.currentIndex() 
            if sessionIsBeeingLoaded:
                self.sessionIsBeeingLoaded = True
            self.combo.clear() 
            self.combo.addItems(["{} ({} x {})".format(fileName,*self.mC.data.getDataFrameShape(dataID)[0]) for dataID, fileName in self.dfs.items()])
            
            if specificIndex is not None and isinstance(specificIndex,int) and specificIndex < len(self.dfs):
                self.combo.setCurrentIndex(specificIndex)
            elif remainLastSelection:
                self.combo.setCurrentIndex(lastIndex)
            elif selectLastDf:
                self.combo.setCurrentIndex(len(dfs)-1)
    
    def getDfIndex(self):
        "Returns the index of the data frame that is selected."
        return self.combo.currentIndex()

    def updateColumnState(self,columnNames, newState = False):
        "The column state indicates if the column is used in the graph or not (bool)"
        for treeView in self.dataHeaders.values():
            treeView.setColumnState(columnNames,newState)
  
    def sendToThread(self, funcProps, addSelectionOfAllDataTypes = False, addDataID = False):
        "Sends a certain request to background thread."
       
        if addSelectionOfAllDataTypes:
            funcProps = self.addSelectionOfAllDataTypes(funcProps)
        if addDataID and "kwargs" in funcProps:
            funcProps["kwargs"]["dataID"] = self.dataID
        self.mC.sendRequestToThread(funcProps)    

    def openMergeDialog(self,e=None):
        "Open a merge dialog."
        dlg = ICDMergeDataFrames(mainController = self.mC)
        dlg.exec()

    def openCorrelateDialog(self,e=None):
        ""
        dlg = ICCorrelateDataFrames(mainController=self.mC)
        dlg.exec()

    def openFeatureCorrelateDialog(self,e=None):
        ""
        dlg = ICCorrelateFeatures(mainController=self.mC)
        dlg.exec()

    def openRenameDialog(self,e=None):
        ""
        dataID = self.mC.getDataID()
        oldName = self.mC.data.getFileNameByID(dataID)
        dlg = AskStringMessage(q="Provide new name for the data frame: {}".format(oldName))
        if dlg.exec():
            funcProps = {"key": "data::renameDataFrame", "kwargs":{"dataID":dataID,"fileName":dlg.state}}
            self.mC.sendRequestToThread(funcProps)

    def openSGCCADialog(self,e=None):
        ""
        # dlg = ICMultiBlockSGCCA(mainController = self.mC)
        # dlg.exec()

    def openProteinPeptideView(self,e=None):
        ""
        dlg = ICProteinProteinView(mainController=self.mC)
        dlg.exec()


    def showMenu(self,e=None):
        "Creates the context menu."
        try:
            #remove focus on button
            sender = self.sender()
            if hasattr(sender,"mouseLostFocus"):
                sender.mouseLostFocus()

            menus = createSubMenu(subMenus=["Grouping .. ","Data frames .. ","Proteomics Toolkit"])#,"Multi block analysis .."
            groupingNames = self.mC.grouping.getNames()
            groupSizes = self.mC.grouping.getSizes()

            if len(groupingNames) > 0:
                # add delete option
                

                for groupingName in groupingNames:
                    menuItemName = "{} ({})".format(groupingName,groupSizes[groupingName])
                    action = menus["Grouping .. "].addAction(menuItemName)#
                    action.triggered.connect(lambda _,groupingName = groupingName : self.updateGrouping(groupingName))
                    


            if self.mC.data.hasData():

                # ugly menu - need to be changed 
                action = menus["Data frames .. "].addAction("Rename")
                action.triggered.connect(self.openRenameDialog)
                #copy dataframe
                action = menus["Data frames .. "].addAction("Copy")
                action.triggered.connect(self.copyDataset)
                ## remove clipping
                action = menus["Data frames .. "].addAction("Reset clipping")
                action.triggered.connect(self.resetClipping)
                #add data frame menu
                action = menus["Data frames .. "].addAction("Merge")
                action.triggered.connect(self.openMergeDialog)

                # action = menus["Data frames .. "].addAction("Correlate")
                # action.triggered.connect(self.openCorrelateDialog)

                action = menus["Data frames .. "].addAction("Correlate features")
                action.triggered.connect(self.openFeatureCorrelateDialog)
                

                menus["Grouping .. "].addSeparator()
                action = menus["Grouping .. "].addAction("Add")
                action.triggered.connect(self.dataHeaders["Numeric Floats"].table.createGroups)
                
                groupingMenus = {}
                if self.mC.grouping.groupingExists():
                    
                    fnMapper = {"Delete ..":self.deleteGrouping,"Rename ..":self.renameGrouping,"Edit ..":self.editGrouping}
                    for menuName in ["Delete ..","Rename ..","Edit .."]:
                        groupingMenus[menuName] = createMenu(menuName)
                        menus["Grouping .. "].addMenu(groupingMenus[menuName])
                        for groupingName in groupingNames:
                            action = groupingMenus[menuName].addAction(groupingName)
                            action.triggered.connect(lambda _, groupingName = groupingName, menuName = menuName : fnMapper[menuName](groupingName))
                    groupingMenus["Export .."] = createMenu("Export ..")

                    groupingMenus["Export .."].addAction("to json", self.exportGrouping) 
                    #groupingMenus["Export .."].addAction("to json (MitoCube)")
                    menus["Grouping .. "].addMenu(groupingMenus["Export .."])

                groupingMenus["Load .."] = createMenu("Load ..")
                menus["Grouping .. "].addMenu(groupingMenus["Load .."])
                groupingMenus["Load .."].addAction("from json", self.importGrouping) 

            action = menus["Proteomics Toolkit"].addAction("Create Sample List")
            action.triggered.connect(self.createSampleList)

            # action = menus["Proteomics Toolkit"].addAction("Protein/Peptide View")
            # action.triggered.connect(self.openProteinPeptideView)
            if False:#self.mC.data.hasTwoDataSets():
                action = menus["Multi block analysis .."].addAction("SGGCA",self.openSGCCADialog)
            
            senderGeom = self.sender().geometry()
            bottomLeft = self.mapToGlobal(senderGeom.bottomLeft())

            menus["main"].exec(bottomLeft)
        except Exception as e:
            print(e)
    
    def exportGrouping(self,*args,**kwargs):
        ""
        funcKey = {"kwargs":{"groupingNames":[]},"key":"groupings:exportGroupingToJson"}
        funcKey = self.mC.askForGroupingSelection(funcKey,numericColumnsInKwargs=False,title="Choose groupings for export.",kwargName="groupingNames")
        
        if len(funcKey["kwargs"]["groupingNames"]) > 0:
            baseFilePath = os.path.join(self.mC.config.getParam("WorkingDirectory"),"ICGroupings")
            fname,_ = QFileDialog.getSaveFileName(self, 'Save file', baseFilePath,"Json files (*.json)")
            if fname:
                funcKey["kwargs"]["filePath"] = fname
                self.mC.sendRequestToThread(funcKey,*args,**kwargs)
            #self.exportGroupingToJson()

    def importGrouping(self,*args,**kwargs):
        "Reads a grouping."
        funcKey = {"key":"groupings:loadGroupingFromJson","kwargs":{}}
        fname,_ = QFileDialog.getOpenFileName(self,'Load json grouping',self.mC.config.getParam("WorkingDirectory"),"Json files (*.json)")   #getSaveFileName(self, 'Save file', baseFilePath,)
        if fname:
            funcKey["kwargs"]["filePath"] = fname
            self.mC.sendRequestToThread(funcKey,*args,**kwargs)
    
    def deleteGrouping(self,groupingName,*args,**kwargs):
        "Remove a grouping."
        key = "grouping::deleteGrouping"
        kwargs = {"groupingName":groupingName}
        self.mC.sendRequestToThread({"key":key,"kwargs":kwargs})

    def editGrouping(self,groupingName,*args,**kwargs):
        ""
        self.dataHeaders["Numeric Floats"].table.createGroups(loadGrouping = True, groupingName = groupingName)

    def renameGrouping(self,groupingName,*args,**kwargs):
        ""
        dlg = AskStringMessage(q="Provide new name for the grouping: {}".format(groupingName))
        if dlg.exec():

            key = "grouping::renameGrouping"
            kwargs = {"groupingName":groupingName,"newGroupingName":dlg.state}
            self.mC.sendRequestToThread({"key":key,"kwargs":kwargs})      

    def copyDataset(self,*args,**kwargs) -> None:
        "Copy a dataset and add to the data treeview"
        dataID = self.mC.getDataID() 
        key = "data::copyDataset"
        funcProps = {"key" : key, "kwargs" : {"dataID" : dataID}}
        self.mC.sendRequestToThread(funcProps)

    def resetClipping(self,*args,**kwargs):
        ""
        self.mC.mainFrames["data"].qS.resetView(resetClipping=False) #clipping will be done here, might be due to dataID diff
        key = "data::resetClipping"
        kwargs = {"dataID":self.mC.getDataID()}
        self.mC.sendRequestToThread({"key":key,"kwargs":kwargs})

    def createSampleList(self,e=None):
        "This is a proteomic-xcalibur functionality."
        dlg = ICSampleListCreater(mainController=self.mC)
        dlg.exec()

    def updateGrouping(self, groupingName):
        "Updates Grouping"
        self.mC.grouping.setCurrentGrouping(groupingName = groupingName)
        groupedItems = self.mC.grouping.getGroupItems(groupingName)
        treeView = self.mC.getTreeView("Numeric Floats")
        treeView.setGrouping(groupedItems,groupingName)
      
