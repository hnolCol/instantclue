from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

#internal imports
from .ICCollapsableFrames import CollapsableFrames
from .buttonDesigns import DataHeaderButton, ViewHideIcon, FindReplaceButton, ResetButton, BigArrowButton, LabelLikeButton, ICStandardButton
from .ICDataTreeView import DataTreeView
from ..dialogs.ICDFindReplace import FindReplaceDialog
from ..dialogs.ICMultiBlockSGCCA import ICMultiBlockSGCCA
from ..dialogs.ICDMergeDataFrames import ICDMergeDataFrames
from ..custom.warnMessage import WarningMessage
from ..utils import WIDGET_HOVER_COLOR, HOVER_COLOR, INSTANT_CLUE_BLUE, getStandardFont, createMenu, createSubMenu


#external imports
import pandas as pd
from collections import OrderedDict


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
    
    def __init__(self, parent=None, sendToThreadFn = None, dfs = OrderedDict(), mainController = None):
        super(CollapsableDataTreeView, self).__init__(parent)

        self.sendToThreadFn = sendToThreadFn
        self.mC = mainController
        self.dfs = dfs #OrderedDict keys: dataID, values: names
        self.preventReset = False
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
        #add function
        self.findReplaceButton.clicked.connect(self.findAndReplace)
        #set up hide shortcuts
        self.hideSC = ViewHideIcon(self)
        self.hideSC.clicked.connect(self.hideShortCuts)
        #export 
        self.exportButton = BigArrowButton(self,tooltipStr="Export selected data to txt file.", buttonSize=(15,15))
        self.exportButton.setContextMenuPolicy(Qt.CustomContextMenu)
        self.exportButton.clicked.connect(self.exportData)
        self.exportButton.customContextMenuRequested.connect(self.exportMenu)
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
            self.dataHeaders[header] = DataTreeView(self, sendToThreadFn = self.sendToThread, tableID = header, mainController=self.mC)
            frame = {"title":header,
                     "open":False,
                     "fixedHeight":False,
                     "height":0,
                     "layout":self.dataHeaders[header].layout()}
            frameWidgets.append(frame)
        self.frames.addCollapsableFrame(frameWidgets, 
                                        closeColor = "#ECECEC", 
                                        openColor = "#ECECEC",
                                        dotColor = INSTANT_CLUE_BLUE,
                                        hoverColor = HOVER_COLOR,
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

    def addDataFrame(self, dataID, dataFrameName):
        "Add a new data frame to the combobox"
        if dataID not in self.dfs:
            self.dfs[dataID] = dataFrameName
            self.updateDfs()
            self.dfSelectionChanged(len(self.dfs)-1)
        
    def reportMenuRequest(self,dataType, menuPosition):
        ""
        #print(dataType, menuPosition)
        if self.dataID is not None:
            sender = self.sender()
            if hasattr(sender,"loseFocus"):
                sender.loseFocus()
            menus = createSubMenu(subMenus=dataTypeMenu)

            for menuItem in menuFuncs:
                action = menus[menuItem["subM"]].addAction(menuItem["name"])
                action.triggered.connect(getattr(self.dataHeaders[dataType],menuItem["funcKey"]))

            menus["main"].exec_(menuPosition)



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
        if dataID is not None and self.sendToThreadFn is not None and self.dataID != dataID:
            #send back to main
            
            funcProps = {
                "key":"data::getColumnNamesByDataID",
                "kwargs":{"dataID":dataID}}
            
            self.dataID = dataID
            self.mC.mainFrames["data"].qS.resetView(updatePlot=False)
            self.mC.mainFrames["data"].liveGraph.clearGraph()
            self.updateDataIDInTreeViews()
            self.sendToThreadFn(funcProps)
           

    def deleteData(self,e=None):
        ""
        self.mC.mainFrames["data"].deleteData()

    def exportData(self,e=None):
        ""
        self.mC.mainFrames["data"].exportData()
        
    def exportMenu(self,e=None):
        ""
        menu = createMenu()
        
        for fileFormat, actionName in [("txt","Tab del. txt"),
                                        ("xlsx", "Excel file"),
                                        ("json", "Json file"),
                                        ("md","Markdown file")]:

            action = menu.addAction(actionName)
            action.triggered.connect(lambda _, txtFileFormat = fileFormat: self.mC.mainFrames["data"].exportData(txtFileFormat))
        senderGeom = self.sender().geometry()
        topLeft = self.mapToGlobal(senderGeom.bottomLeft())
        menu.exec_(topLeft) 

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
            senderGeom = self.sender().geometry()
            topLeft = self.parent().mapToGlobal(senderGeom.bottomLeft())
            frd = FindReplaceDialog(self.mC)
            frd.setGeometry(topLeft.x(),topLeft.y(),200,150)
            if frd.exec_():
                funcKey = "data::replace"
                fS = frd.findStrings 
                rS = frd.replaceStrings
                specificColumnSelected = frd.specificColumnSelected #bool
                selectedIndex = frd.selectedColumnIndex
                dataType = frd.selectedDataType # selected data type
                if selectedIndex == 0: #this means complete selection, save if colum header is by change same in data

                    specificColumn = self.getSelectedColumns(dataType=dataType)
                    if len(specificColumn) == 0:
                        w = WarningMessage(iconDir = self.mC.mainPath, infoText = "No selected columns found in selected datat type: {}".format(dataType))
                        w.exec_()
                        return
                    
                  #  specificColumn = frd.selectedColumn
                elif specificColumnSelected:
                    specificColumn = [frd.selectedColumn]
                else:
                    specificColumn = None

                funcProps = {"key":funcKey,"kwargs":{"findStrings":fS,"replaceStrings":rS,"specificColumns":specificColumn,"dataID":self.mC.getDataID(),"dataType":dataType}}
                self.sendToThread(funcProps)
        except Exception as e:
            print(e)

    def hideShortCuts(self):
        "User can hide/show shortcuts"
        self.sender().stateChanged()
        for treeView in self.dataHeaders.values():
            treeView.hideShowShortCuts()

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

    def updateDfs(self, dfs = None, selectLastDf = True, remainLastSelection = False):
        ""
        if dfs is not None:
            self.dfs = dfs
            if remainLastSelection:
                lastIndex = self.combo.currentIndex() 
            self.combo.clear() 

            self.combo.addItems(list(self.dfs.values()))
            if remainLastSelection:
                self.combo.setCurrentIndex(lastIndex)
            elif selectLastDf:
                self.combo.setCurrentIndex(len(dfs)-1)

    def updateColumnState(self,columnNames, newState = False):
        "The column state indicates if the column is used in the graph or not (bool)"
        for treeView in self.dataHeaders.values():
            treeView.setColumnState(columnNames,newState)
  
    def sendToThread(self, funcProps, addSelectionOfAllDataTypes = False, addDataID = False):
        ""
        if hasattr(self,"sendToThreadFn"):
            if self.sendToThreadFn is not None:
                if addSelectionOfAllDataTypes:
                    funcProps = self.addSelectionOfAllDataTypes(funcProps)
                if addDataID and "kwargs" in funcProps:
                    funcProps["kwargs"]["dataID"] = self.dataID
                self.sendToThreadFn(funcProps)
              

    def openMergeDialog(self,event=None):
        ""
        dlg = ICDMergeDataFrames(mainController = self.mC)
        dlg.exec_()
    
    def openSGCCADialog(self,event=None):
        ""
        # dlg = ICMultiBlockSGCCA(mainController = self.mC)
        # dlg.exec_()

    def showMenu(self,event=None):
        ""
        try:
            #remove focus on button
            sender = self.sender()
            if hasattr(sender,"mouseLostFocus"):
                sender.mouseLostFocus()

            menus = createSubMenu(subMenus=["Grouping .. ","Data frames .. "])#,"Multi block analysis .."
            groupingNames = self.mC.grouping.getNames()
            groupSizes = self.mC.grouping.getSizes()
            if len(groupingNames) > 0:
                for groupingName in groupingNames:
                    menuItemName = "{} ({})".format(groupingName,groupSizes[groupingName])
                    action = menus["Grouping .. "].addAction(menuItemName)#
                    action.triggered.connect(lambda _,groupingName = groupingName : self.updateGrouping(groupingName))
            elif self.mC.data.hasData():
                action = menus["Grouping .. "].addAction("Add Grouping")
                action.triggered.connect(self.dataHeaders["Numeric Floats"].table.createGroups)

            if self.mC.data.hasData():
                #add data frame menu
                action = menus["Data frames .. "].addAction("Merge")
                action.triggered.connect(self.openMergeDialog)

            if False:#self.mC.data.hasTwoDataSets():
                action = menus["Multi block analysis .."].addAction("SGGCA",self.openSGCCADialog)
                
            
            senderGeom = self.sender().geometry()
            bottomLeft = self.mapToGlobal(senderGeom.bottomLeft())

            menus["main"].exec_(bottomLeft)
        except Exception as e:
            print(e)
    
    def updateGrouping(self, groupingName):
        ""
        self.mC.grouping.setCurrentGrouping(groupingName = groupingName)
        groupedItems = self.mC.grouping.getGroupItems(groupingName)
        treeView = self.mC.getTreeView("Numeric Floats")
        treeView.setGrouping(groupedItems,groupingName)
      
