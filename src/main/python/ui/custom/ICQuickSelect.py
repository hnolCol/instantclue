from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

from ..delegates.quickSelectDelegates import DelegateColor, DelegateSize
from .buttonDesigns import ArrowButton, ResetButton, CheckButton, MaskButton, AnnotateButton, SaveButton, BigArrowButton, SmallColorButton
from ..dialogs.quickSelectDialog import QuickSelectDialog
from ..utils import createMenu, createSubMenu, getMessageProps, HOVER_COLOR, getStandardFont, legendLocations
import os
import pandas as pd
import numpy as np

import pickle

class QuickSelect(QWidget):
    def __init__(self,sendToThreadFn = None, mainController = None, parent=None):
        
        super(QuickSelect, self).__init__(parent)
        self.__controls()
        self.__layout()
        self.__connectEvents()

        self.setAcceptDrops(True)
        self.sendToThreadFn = sendToThreadFn
        self.favSelection = FavoriteSelectionCollection(mainController)
        self.mC = mainController
        self.quickSelectProps = {}
        self.hoverIdx = {}
        self.lastIdx = None
        #set up default
        self.setAnnotateMode()

    def __controls(self):

        self.__setupTable()
        self.searchLineEdit = QLineEdit(self)
        self.searchLineEdit.setPlaceholderText("Search ...")
        self.searchLineEdit.textChanged.connect(self.model.search)
    
        self.annotateButton = AnnotateButton(self,tooltipStr="Annotate Selection. If this is already selected, clicking again will set the mode to 'mask'")
        self.maskButton = MaskButton(self,tooltipStr="Mask unchecked items in data. To remove mask, select 'Annotate Select' (left) or click again (toggle)")
 
        self.sortAscendingButton = ArrowButton(self, tooltipStr="Sort Quick Select in ascending order.\nClick again to restore raw data order.")
        self.sortDescendingButton = ArrowButton(self,direction="down", tooltipStr="Sort Quick Select in descending order.\nClick again to restore raw data order.")

                
        self.colorButton = SmallColorButton(tooltipStr="Set color by categorical column or clustering.")

        self.saveButton = BigArrowButton(self,
                            direction="down" ,
                            tooltipStr="Save Selection.\nCan be loaded and applied to any dataset by matching values.",
                            buttonSize=(15,15))
        self.loadButton = BigArrowButton(self,
                            direction="up" ,
                            tooltipStr="Load saved selection.\nMatches will be performed by string value not by index.\nThis means that if there are multiple matches, all with be selected.",
                            buttonSize=(15,15))
        self.checkedLabels = CheckButton(self,tooltipStr="Shows (un)checked items only.")
        self.resetButton = ResetButton(self,tooltipStr="Reset View")


    def __connectEvents(self):
    
        self.sortDescendingButton.clicked.connect(lambda e : self.model.sort(how="descending"))
        self.sortDescendingButton.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sortDescendingButton.customContextMenuRequested.connect(lambda e: self.openSortMenu(sortHow="descending"))

        self.sortAscendingButton.clicked.connect(self.model.sort)
        self.sortAscendingButton.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sortAscendingButton.customContextMenuRequested.connect(self.openSortMenu)
        self.resetButton.clicked.connect(self.resetView)
        self.checkedLabels.clicked.connect(self.showCheckedLabels)
        self.maskButton.clicked.connect(self.setMaskMode)
        self.annotateButton.clicked.connect(self.setAnnotateMode)
        self.saveButton.clicked.connect(self.saveSelection)
        self.loadButton.clicked.connect(self.loadSelection)
        self.colorButton.clicked.connect(self.openColorMenu)


    def __setupTable(self):
        
        self.table = QuickSelectTableView(self)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.model = QuickSelectModel(parent=self)
        self.table.setModel(self.model)
        
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch) 
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.Fixed)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.resizeColumns()
        
        self.table.setItemDelegateForColumn(1,DelegateColor(self.table))
        self.table.setItemDelegateForColumn(2,DelegateSize(self.table))
        
    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        hLayout = QHBoxLayout()
        hLayout.addWidget(self.searchLineEdit)
        hLayout.addWidget(self.annotateButton)
        hLayout.addWidget(self.maskButton )
        hLayout.addWidget(self.colorButton)
        hLayout.addWidget(self.sortAscendingButton)
        hLayout.addWidget(self.sortDescendingButton)
        hLayout.addWidget(self.loadButton)
        hLayout.addWidget(self.saveButton)
        hLayout.addWidget(self.resetButton)
        hLayout.addWidget(self.checkedLabels)
        hLayout.setContentsMargins(0,0,0,0)
        hLayout.setSpacing(1)
        self.layout().addLayout(hLayout)
        self.layout().addWidget(self.table)
        self.layout().setSpacing(1)
        self.layout().setContentsMargins(1,1,1,1)
    

    def sortOptions(self,event=None):
        ""
        pass

    def leaveEvent(self,event):
        ""
        exists, graph = self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.updateFigure.emit()

    def dragEnterEvent(self,event):
        ""
        event.accept()

    def dropEvent(self,event):
        ""
        event.accept()
        dfg = QuickSelectDialog(mainController=self.mC)

        try: 
            eventPoint = self.mapToGlobal(event.pos())
            dfg.setGeometry(eventPoint.x()-50,eventPoint.y(),100,100)
            if dfg.exec_():
                props = dfg.getProps()
                columnNames = self.mC.mainFrames["data"].getDragColumns()
                props["columnName"] = columnNames.iloc[0]
                self.updateQuickSelectData(columnNames.iloc[0],props)

        except Exception as e:
            print(e)
   
    def updateQuickSelectData(self,columnName,filterProps,dataID = None,):
        ""
        #rest quickSelect Table
        self.mC.resetQuickSelectTable.emit()
        #create kwars/func for Thread
        funcProps = dict()
        if dataID is None:
            dataID = self.mC.mainFrames["data"].getDataID()
        
        if "sep" in filterProps:
            self.mC.config.setParam("quick.select.separator",filterProps["sep"])
            if filterProps["sep"] in ["tab","space"]:
                sepString = filterProps["sep"]
                filterProps["sep"] = "\t" if sepString == "tab" else " "
        else:
            filterProps["sep"] = self.mC.config.getParam("quick.select.separator")
        funcProps["key"] = "data::updateQuickSelectData"
        funcProps["kwargs"] = {"filterProps":filterProps,"dataID":dataID}
        #save filter options 
        self.quickSelectProps["columnName"] = columnName
        self.quickSelectProps["dataID"] = dataID
        self.quickSelectProps["filterProps"] = filterProps
        self.sendToThreadFn(funcProps)



    def showCheckedLabels(self):
        ""
        self.sender().toggleState()
        self.table.model().layoutAboutToBeChanged.emit()
        self.model.showCheckedLabels()
        self.table.model().layoutChanged.emit()

    def addData(self,X):
        ""
        
        self.table.model().layoutAboutToBeChanged.emit()
        self.table.model().setNewData(X)
        self.table.model().layoutChanged.emit()


    def setMaskMode(self,event=None):
        ""
        if hasattr(self,"selectionMode") and self.selectionMode == "mask":
            self.setAnnotateMode()
        else:
            self.selectionMode = "mask"
            self.annotateButton.setState(False)
            self.maskButton.setState(True)
            self.resetGraphItems()
            self.updateMode() 

    def setAnnotateMode(self,event=None):
        ""
        if hasattr(self,"selectionMode") and self.selectionMode == "annotate":
            self.setMaskMode()
        else:
            self.selectionMode = "annotate"
            #remove clipping
            self.resetClipping()
            self.annotateButton.setState(True)
            self.maskButton.setState(False)
            self.updateMode() 

            

    def setCheckStateByDataIndex(self,dataIndex):
        ""
        if len(self.quickSelectProps) == 0:
            return
        #user can only check if mode is raw
        if dataIndex is not None and self.model.rowCount() > 0 \
                and self.quickSelectProps["filterProps"]["mode"] == "raw":

            self.table.model().setCheckStateByDataIndex(dataIndex)
            self.table.model().completeDataChanged()

    def updateMode(self):
        ""
        if len(self.quickSelectProps) > 0 and self.model.getNumberOfSelectedRows() > 0:
            self.updateDataSelection()

    def updateDataSelection(self, checkedLabels = None):
        ""
        
        if self.model._inputLabels.size == 0:
            return
        try:
            if checkedLabels is None:
                    checkedLabels = self.model.getCheckedData()
            self.quickSelectProps["checkedLabels"] =  checkedLabels
            
            if self.selectionMode == "mask":

                funcProps = self.getMaskingProps()

            else:

                funcProps = self.getAnnotateProps() 
            self.mC.sendRequestToThread(funcProps)
           
            return True
        except Exception as e:
            print(e)
   

    def getAnnotateProps(self, ignoreUserSelection = False):
        ""
        funcProps = {}
        funcProps["key"] = "data::getColorDictsByFilter"
        funcProps["kwargs"] = self.quickSelectProps.copy()
        if not ignoreUserSelection:
            funcProps["kwargs"]["checkedSizes"] = self.model.getCheckedSizes()
            funcProps["kwargs"]["userColors"] = self.model.getUserDefinedColors()
        if self.quickSelectProps["filterProps"]["mode"] == "raw":
            funcProps["kwargs"]["checkedDataIndex"] = self.model.checkedLabels
        
        return funcProps

    def getMaskingProps(self):
        ""    
        funcProps = {}
        funcProps["key"] = "data::setClippingMask"
        funcProps["kwargs"] = self.quickSelectProps.copy()
        if self.quickSelectProps["filterProps"]["mode"] == "raw":
            funcProps["kwargs"]["checkedDataIndex"] = self.model.checkedLabels
        
        return funcProps

    def getCurrentLabel(self):
        ""
        return self.table.getCurrentHighlightLabel()

    def getFilterMode(self):
        ""
        return self.quickSelectProps["filterProps"]["mode"]

    def getQuickSelectColumn(self):
        ""
        return self.quickSelectProps["columnName"]


    def openSortMenu(self,event=None, sortHow = "ascending"):
        ""
        
        menu = createMenu()
        menu.addAction("By color", lambda sortHow = sortHow:self.model.sortByColor(how=sortHow))
        menu.addAction("By size")

        senderGeom = self.sender().geometry()
        topLeft = self.mapToGlobal(senderGeom.bottomLeft())
        menu.exec_(topLeft) 
        self.sender().mouseLostFocus()

    def openColorMenu(self,event=None):
        ""
        if self.model._labels.size > 0:
            if self.quickSelectProps["filterProps"]["mode"] == "raw":
                try:
                    menus = createSubMenu(subMenus=["By catgeorical column","By clustering"])
                    action = menus["By clustering"].addAction("k-means")
                    action.triggered.connect(self.colorLabelsByCluster)
                    dataID = self.mC.mainFrames["data"].getDataID() 
                    categoricalColumns = self.mC.data.getCategoricalColumns(dataID)
                    for catColumn in categoricalColumns:
                        action = menus["By catgeorical column"].addAction(catColumn)
                        action.triggered.connect(self.colorLabelsByCategoricalColumn)
                    senderGeom = self.sender().geometry()
                    topLeft = self.mapToGlobal(senderGeom.bottomLeft())
                    menus["main"].exec_(topLeft) 
                    
                except Exception as e:
                    print(e)
            else:
                self.mC.sendMessageRequest(getMessageProps("Information","Coloring is only available if raw data instead of unique data were loaded."))
        else:
             self.mC.sendMessageRequest(getMessageProps("Information","Drag & Drop a column to Quick Select Widget."))
    
    def colorLabelsByCluster(self,event=None):
        ""
        pass

    def colorLabelsByCategoricalColumn(self,event=None,categoricalColumn = None):
        #print("cat column")
        if categoricalColumn is None:
            categoricalColumn = self.sender().text()
            dataID = self.mC.mainFrames["data"].getDataID() 
            colorData = self.mC.data.colorManager.getColorByDataIndex(dataID, categoricalColumn)
            self.model.setColorSeries(colorData)
            self.model.completeDataChanged()
         

    def updateColorsAndSizes(self,checkedColors=None,checkedSizes=None):
        ""
        if checkedSizes is not None:
            
            self.model.setSizes(checkedSizes)

        if checkedColors is not None:
            
            self.model.setCheckedColors(checkedColors)
        
        self.table.model().completeDataChanged()

    
    def setHighlightIndex(self,dataIndex):
        ""
        if len(self.quickSelectProps) == 0:
            return
        if self.selectionMode != "annotate":
            return
        if self.searchLineEdit.text() != "":
            self.searchLineEdit.setText("")

        if self.checkedLabels.getState():
            self.checkedLabels.setState(False)
            self.table.model().showCheckedOnly = False
        
        if self.quickSelectProps["filterProps"]["mode"] == "unique" and dataIndex is not None:
            tableDataIndex = None
            if "filterProps" in self.quickSelectProps:
                #get split string from users selection
                splitString = self.quickSelectProps["filterProps"]["sep"]
                #get column name 
                columnName = self.quickSelectProps["columnName"]
                #retrieve dataID
                dataID = self.mC.mainFrames["data"].getDataID() 
                #get the value and split it on selected split string
                values = self.mC.data.getDataValue(dataID,columnName,dataIndex,splitString=splitString)
                #if dataID or dataIndex not found, None is returned
                tableDataIndex = self.model.getIndexForMatches(values)

        else:
            tableDataIndex = dataIndex
        #if index is None and complete data are anyway shown 
        #return None
        if tableDataIndex is None and self.model.isCompleteDataShown():
            return
        #prepare tabl that layout will be cange
        #this is important if the user scrolled down. (no view of highlighted row)
        self.table.model().layoutAboutToBeChanged.emit()
        #set highlight index
        self.table.model().showHighlightIndex(tableDataIndex, updateData = False)
        #notify model/table that layout changed
        self.table.model().layoutChanged.emit()
        #update data 
        self.model.completeDataChanged()

    def exportSelectionToData(self,event=None):
        ""
        filterMode = self.quickSelectProps["filterProps"]["mode"]
        selectionData = self.model.getCompleteSelectionData(attachSizes=True)
        selectionData = selectionData.dropna(subset=["checkedValues"])
        selectionData = selectionData.dropna(axis=1,how="all")
        

        if selectionData["checkedValues"].index.size == 0:
                self.mC.sendMessageRequest({"title":"Error ..","message":"No selection made in Quick Select"})
                return  

        if filterMode == "raw":
            funcProps = {}
            funcProps["key"] = "data::joinDataFrame"
            funcProps["kwargs"] = {}
            funcProps["kwargs"]["dataID"] = self.mC.getDataID()
            funcProps["kwargs"]["dataFrame"] = selectionData
            self.sendToThreadFn(funcProps)

        else:
            self.mC.sendMessageRequest({"title":"Error ..","message":"Only supported for raw selection mode."})

    def exportSelectionToClipbard(self,event=None, attachColumns = False):
        ""

        selectionData = self.model.getCompleteSelectionData(attachSizes=True)
        selectionData = selectionData.dropna(subset=["checkedValues"])

        if selectionData["checkedValues"].index.size == 0:
                self.mC.sendMessageRequest({"title":"Error ..","message":"No selection made in Quick Select"})
                return  
        else:
            
            if attachColumns:
                funcProps = {}

                filterMode = self.quickSelectProps["filterProps"]["mode"]
                if filterMode == "raw":
                    funcProps["key"] = "data:copyDataToClipboard"
                    funcProps["kwargs"] = {}
                    funcProps["kwargs"]["attachDataToMain"] = selectionData
                    funcProps["kwargs"]["dataID"] = self.mC.getDataID()

                else:  
                    
                    funcProps["key"] = "data:copyDataFromQuickSelectToClipboard"
                    funcProps["kwargs"] = {}
                    funcProps["kwargs"]["columnName"] = self.quickSelectProps["columnName"]
                    funcProps["kwargs"]["dataID"] = self.mC.getDataID()
                    funcProps["kwargs"]["splitString"] = self.quickSelectProps["filterProps"]["sep"]
                    funcProps["kwargs"]["selectionData"] = selectionData


                self.sendToThreadFn(funcProps)

            else:

                selectionData.to_clipboard()

            self.mC.sendMessageRequest({"title":"Done ..","message":"Selection copied to clipboard."})

    def getCurrentDataIdx(self, dataIndex, searchString = None):
        ""
        filterMode = self.quickSelectProps["filterProps"]["mode"]
        if filterMode == "raw":
            hoverIdx = np.array([dataIndex])
        else:  

            if searchString is None:
                searchString = self.table.getCurrentHighlightLabel()

            if searchString in self.hoverIdx:
                hoverIdx = self.hoverIdx[searchString]
            else:
                columnName = self.quickSelectProps["columnName"]
                dataID = self.quickSelectProps["dataID"]
                splitString = self.quickSelectProps["filterProps"]["sep"]
                hoverIdx = self.mC.data.columnRegExMatches(dataID,[columnName],searchString,splitString)
                if hoverIdx.empty:
                    return
                self.hoverIdx[searchString] = hoverIdx

        return hoverIdx

    def saveSelection(self,event=None, selectionData = None):
        ""

        try:

            if self.model.rowCount() == 0:
                self.mC.sendMessageRequest({"title":"Error ..","message":"No data in Quick Select.\nDrag & drop categorical column."})
                return   
            if selectionData is None: 
                selectionData = self.model.getCompleteSelectionData()
                selectionData = selectionData.dropna(subset=["checkedValues"])

            if selectionData["checkedValues"].index.size == 0:
                self.mC.sendMessageRequest({"title":"Error ..","message":"No selection made in Quick Select"})
                return   

            text, ok = QInputDialog.getText(self, 'Save Quick Selection', 'Enter name of selection:')
            if ok:
                selectName = text
            else:
                return
            response = self.favSelection.add(selectionData,selectName)
            if isinstance(response,str):
                self.mC.sendMessageRequest({"title":"Error..","message":response})
            else:
                self.mC.sendMessageRequest({"title":"Saved ..","message":"Selection saved.{}".format(selectName)})
        except Exception as e:
            print(e)
    
    def loadSelection(self,event = None, selectionName=None):
        ""
        
        try:
            if selectionName is None:
                savedSelections = self.favSelection.getSavedSelections()
                if len(savedSelections) == 0:
                    return
                #set up menu
                loadMenu = createMenu()
                for f in savedSelections:
                    loadMenu.addAction(f)
                #find bottom left corner
                senderGeom = self.sender().geometry()
                topLeft = self.parent().mapToGlobal(senderGeom.bottomLeft())
                #set sender status 
                self.sender().mouseOver = False
                #cast menu
                action = loadMenu.exec_(topLeft)
                #if action not None(e.g. item was not closed)
                if action is not None:
                    selectionName = action.text()
            
            #retrieve selection data
            selectionData = self.favSelection.load(selectionName)
            #update model
            self.model.readFavoriteSelection(selectionData)
        except Exception as e:
            print(e)

    def deleteSelection(self,selectionName):
        ""
        response = self.favSelection.delete(selectionName)
        if isinstance(response,str):
            self.mC.sendMessageRequest({"title":"Error..","message":response})
       
    def readSelection(self, readType = "Text/CSV file"):
        ""
        selectionData = dict() 
        if readType == "Clipboard":
            data = pd.read_clipboard(sep="\t",header=None)
            data = data.values.flatten()
        elif readType == "Text/CSV file":
            fname,_ = QFileDialog.getOpenFileName(self, 'Open file', 
                    "Text/CSV files (*.txt *.csv)")
            #if user cancels file selection, return function
            if fname:
                df = self.mC.data.addDataFrameFromTxtFile(fname, fileName='', returnPlainDf=True)
                #if loading did not work
                #a dict with error message is returned
                if isinstance(df,dict):
                    self.mC.sendMessageRequest(df)
                elif isinstance(df,pd.DataFrame):
                    data = df.values.flatten()
                else:
                    self.mC.sendMessageRequest(getMessageProps("Error ..","There was an error loading the file."))
            else:
                return
        
        selectionData["checkedValues"] = pd.Series(data)
        selectionData["checkedColors"] = pd.Series()
        selectionData["userDefinedColors"] = pd.Series() 
        
        self.model.readFavoriteSelection(selectionData)

    def highlightDataInPlotter(self,dataIndex):
        ""
        try:
            hoverIdx = None
            if self.lastIdx != dataIndex:
                exists, graph = self.mC.getGraph()
                if exists:
                    hoverIdx = self.getCurrentDataIdx(dataIndex)
                    if hoverIdx is not None:
                        graph.setHoverData(hoverIdx)
                if hoverIdx is not None:
                    self.mC.mainFrames["data"].liveGraph.updateGraphByIndex(hoverIdx)

                self.lastIdx = dataIndex
            

        except Exception as e:
            print(e)

    def resetView(self,event=None, updatePlot = True):
        ""
        self.model.resetView()
        if "dataID" in self.quickSelectProps and updatePlot:
            self.resetClipping()
        #reset size and color tables
        
        self.quickSelectProps.clear()
        self.resetGraphItems() 

    def resetGraphItems(self):
        ""
        self.mC.resetQuickSelectTable.emit()
        exists, graph = self.mC.getGraph()
        if exists:
            graph.resetQuickSelectArtists()
        

    def getSizeAndColorData(self):
        ""
        c = self.model.checkedColors
        s = self.model.checkedSizes
        checkedLabels = self.model.checkedLabels.index[self.model.checkedLabels]

        data = pd.concat([c,s],axis=1)
        data.columns = ["color","size"]
        data["layer"] = -1 
        data.loc[checkedLabels,"layer"] = 1
        data.sort_values(by="layer",ascending=True,inplace=True)#keep selected labels on top
        return data[["color","size"]]
    
    def getDataIndexOfCurrentSelection(self):
        ""
        return self.model.checkedLabels.index[self.model.checkedLabels]

    def hasData(self):
        ""
        return self.model.rowCount() > 0
        

    def resetClipping(self):
        "Resets Clipping"
        if "dataID" in self.quickSelectProps:
            funcProps = {"key":"data::resetClipping","kwargs":{"dataID":self.quickSelectProps["dataID"]}}
            self.sendToThreadFn(funcProps)


class FavoriteSelectionCollection(object):
    def __init__(self, mainController):
        self.mC = mainController
        self.pathToTemp = os.path.join(self.mC.mainPath,"quickSelectLists")
        if not os.path.exists(self.pathToTemp):
            os.mkdir(self.pathToTemp)

    def add(self,selectionData,selectionName):
        ""
        pathToFile = self.getPath(selectionName)
        try:
            selectionData.to_csv(pathToFile, index=False,sep="\t")
            return True
        except PermissionError:
            return "Not saved. Permission Error" 
        except:
            return "Not saved Unknown Error"

    def load(self,selectionName):
        ""
        pathToFile = self.getPath(selectionName)
        if os.path.exists(pathToFile):
            selectionData = pd.read_table(pathToFile, sep="\t")
           
            return selectionData
    
    def delete(self,selectionName):
        ""
        if selectionName in self.getSavedSelections():
            pathToFile = self.getPath(selectionName)
            try:
                os.remove(pathToFile)
                return True
            except PermissionError:
                return "Not removed. Permission Error" 
            except:
                return "Not removed. Unknown Error"
            

    def getSavedSelections(self):
        ""
        files = os.listdir(self.pathToTemp)
        files = [f.rsplit(".",1)[0] for f in files if f.endswith(".txt")]
        return files

    def getPath(self, selectionName):
        ""
        return os.path.join(self.pathToTemp,"{}.txt".format(selectionName))


class QuickSelectModel(QAbstractTableModel):
    
    def __init__(self, labels = pd.Series(), parent=None):
        super(QuickSelectModel, self).__init__(parent)
        self.initData(labels)

    def initData(self,labels):
        ""
        self._labels = labels
        self._inputLabels = labels.copy()
        self.setCheckedSeries()
        self.initColorSeries()
        self.setSizeSeries()
        self.setDefaultSize()
        self.lastSearchType = None
        self.showCheckedOnly = False

    def rowCount(self, parent=QModelIndex()):
        ""
        return self._labels.size

    def columnCount(self, parent=QModelIndex()):
        ""
        return 3

    def setCheckedSeries(self):
        ""
        if self.rowCount() == 0:
            self.checkedLabels = pd.Series()
        else:
            self.checkedLabels = pd.Series(np.zeros(shape=self.rowCount()), index=self._inputLabels.index)
            self.checkedLabels = self.checkedLabels.astype(bool)

    def initColorSeries(self):
        ""
        if self.rowCount() == 0:
            self.checkedColors = pd.Series()
            self.userDefinedColors = pd.Series() 
        else:
            self.checkedColors = pd.Series(
                        [self.parent().mC.config.getParam("nanColor")] * self.rowCount(),
                        index=self._inputLabels.index)

    def setColorSeries(self, colorSeries):
        ""
       # self.checkedColors = colorSeries
    
        self.userDefinedColors = colorSeries


    def setSizeSeries(self):
        ""
        if self.rowCount() == 0:
            self.checkedSizes = pd.Series()
        else:
            self.checkedSizes = pd.Series(
                                np.full(
                                shape=self.rowCount(), 
                                fill_value=self.parent().mC.config.getParam("scatterSize")), 
                                index=self._inputLabels.index)
        

    def setDefaultSize(self,size=50):
        ""
        self.defaultSize = size

    def getDataIndex(self,row):
        ""
        return self._labels.index[row]
    
    def getDataByRow(self,row):
        ""
        if row < self._labels.size:
            return self._labels.iloc[row]
        
    def getCheckStateByDataIndex(self,dataIndex):
        ""
        return self.checkedLabels.loc[dataIndex] == 1

    def getCheckStateByTableIndex(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        return self.getCheckStateByDataIndex(dataIndex)

    def setCheckState(self,tableIndex, update=True):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        newState = not self.checkedLabels.loc[dataIndex]
        self.checkedLabels.loc[dataIndex] = newState
        if update:
            self.parent().updateDataSelection()

    def setCheckStateByMultipleIndx(self,dataIndex,update):
        ""
        self.checkedLabels.loc[dataIndex] = True
        #set colors is missing?
        if update:
            self.parent().updateDataSelection()


    def setCheckStateByDataIndex(self,dataIndex,update=True):
        ""
        if isinstance(dataIndex,pd.Int64Index) or isinstance(dataIndex,pd.Series):
            self.setCheckStateByMultipleIndx(dataIndex,update)
        else:
            if dataIndex in self.checkedLabels.index:
                newState = not self.checkedLabels.loc[dataIndex]
                self.checkedLabels.loc[dataIndex] = newState 
                if update:
                    self.parent().updateDataSelection()

    def setCheckedColors(self,checkedColors):
        ""
        self.checkedColors = checkedColors

    def setColor(self,dataIndex,newColor, mode = "add", update=True):
        ""
        if mode == "remove":
            if dataIndex in self.userDefinedColors:
                self.userDefinedColors = self.userDefinedColors.drop(dataIndex)
        elif mode == "add":
            if dataIndex in self.userDefinedColors:
                self.userDefinedColors.loc[dataIndex] = newColor
            else:
                newColorItem = pd.Series([newColor],index=[dataIndex])
                self.userDefinedColors = self.userDefinedColors.append(newColorItem,verify_integrity=True)
        if update:
            self.parent().updateDataSelection()

    def setSizes(self,checkedSizes):
        ""
        self.checkedSizes = checkedSizes

    def setSize(self,dataIndex,newSize, mode= "add"):
        ""
        if mode == "remove":
            if dataIndex in self.checkedSizes.index:
                self.checkedSizes = self.checkedSizes.drop(dataIndex)
        elif mode == "add":
            if dataIndex in self.checkedSizes.index:
                self.checkedSizes.loc[dataIndex] = newSize
            else:
                newSizeItem = pd.Series([newSize],index=[dataIndex])
                self.checkedSizes = self.checkedSizes.append(newSizeItem,verify_integrity=True)
        self.parent().updateDataSelection()

    def getCheckedColors(self):
        ""
        return self.checkedColors

    def getUserDefinedColors(self):
        ""
        return self.userDefinedColors

    def getColor(self,tableIndex):
        ""
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex in self.userDefinedColors.index:
            return self.userDefinedColors.loc[dataIndex]
        if dataIndex in self.checkedColors.index:
            return self.checkedColors.loc[dataIndex]
        else:
            return QColor("lightgrey")
        
    def getCheckedData(self):
        ""
        checkedIndex = self.checkedLabels[self.checkedLabels == 1].index
        return self._inputLabels.loc[checkedIndex]

    def getCompleteSelectionData(self, attachSizes = False):
        ""
        selectionData = dict()
        selectionData["checkedValues"] = self.getCheckedData()
        selectionData["checkedColors"] = self.checkedColors
        selectionData["userDefinedColors"] = self.userDefinedColors
        if attachSizes:
            selectionData["checkSizes"] = self.checkedSizes
        selectionData = pd.DataFrame().from_dict(selectionData)
        return selectionData
    
    def getIndexForMatches(self,dataList):
        ""
        return pd.Series(self._inputLabels.index[self._inputLabels.isin(dataList)])

    def readFavoriteSelection(self, selectionData):
        ""
        if not self._inputLabels.empty:
            checkedData = selectionData["checkedValues"]
            caseSensitive = self.parent().mC.config.getParam("quick.select.case.sensitive")
            if caseSensitive: #make each string lowercase and match then
                lowerStrLabels = self._inputLabels.str.lower()
                lowerStrSelectionData = selectionData["checkedValues"].str.lower()
                boolMatch = lowerStrLabels.isin(lowerStrSelectionData)
            else: #exact matching
                boolMatch = self._inputLabels.isin(checkedData.values)
            if np.any(boolMatch.values):
                dataMatched = self._inputLabels[boolMatch]
                for index, value in  dataMatched.iteritems():
                    if caseSensitive:
                        lowerValue = lowerStrLabels.loc[index]
                        findSavedIndex = lowerStrSelectionData[lowerStrSelectionData.values == lowerValue].index.values[0]
                    else:
                        findSavedIndex = checkedData[checkedData.values == value].index.values[0]
                    if not self.getCheckStateByDataIndex(index):
                        self.setCheckStateByDataIndex(index,update=False)
                    if isinstance(selectionData.loc[findSavedIndex,"userDefinedColors"],str):
                        self.setColor(index,selectionData.loc[findSavedIndex,"userDefinedColors"],update=False)

        self.completeDataChanged()
        self.parent().updateDataSelection()

    def getSize(self,tableIndex):
        ""
        
        dataIndex = self.getDataIndex(tableIndex.row())
        if dataIndex in self.checkedSizes.index:
            return self.checkedSizes.loc[dataIndex]
        else:
            return self.defaultSize

    def getCheckedSizes(self):
        ""
        return self.checkedSizes

    def getNumberOfSelectedRows(self):
        ""
        return np.sum(self.checkedLabels.values)

    def isCompleteDataShown(self):
        ""
        return self._labels.index.size == self._inputLabels.index.size

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

    def data(self, index, role=Qt.DisplayRole): 
        ""
        if not index.isValid(): 
            return QVariant()
        elif role == Qt.DisplayRole and index.column() == 0: 
            return str(self._labels.iloc[index.row()])
        elif role == Qt.CheckStateRole:
            if index.column() != 0:
                return QVariant()
            if self.getCheckStateByTableIndex(index):
                return Qt.Checked
            else:
                return Qt.Unchecked
        elif role == Qt.TextAlignmentRole:
            
            return Qt.AlignVCenter

        elif self.parent().table.mouseOverItem is not None and role == Qt.BackgroundRole and index.row() == self.parent().table.mouseOverItem:
            
            return QColor(HOVER_COLOR)

        elif role == Qt.FontRole and index.column() == 0:

            font = self.getFont()
            if self.getCheckStateByTableIndex(index):
                font.setItalic(True)

            return font


    def getFont(self):
        ""
        return getStandardFont()


    def dataAvailable(self):
        ""
        return self._labels.size > 0 

    def flags(self, index):

        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable 


    def setNewData(self,labels):
        ""
        self.initData(labels)
        self.completeDataChanged()


    def search(self,searchString):
        ""
        self.layoutAboutToBeChanged.emit()
        if self._inputLabels.size == 0:
            return
        if len(searchString) > 0:
            boolMask = self._inputLabels.astype(str).str.contains(searchString,case=False,regex=False)
            self._labels = self._inputLabels.loc[boolMask]
        else:
            self._labels = self._inputLabels.copy()
            self.parent().checkedLabels.setState(False)
            #set button state back to False
            if self.showCheckedOnly:
                self.showCheckedLabels()
                #show check labels includes complete data update
                return
        self.layoutChanged.emit()
        self.completeDataChanged()

    def sort(self, e = None, how = "ascending"):
        ""
        if self._inputLabels.size == 0:
            return
        if self.lastSearchType is None or self.lastSearchType != how:

            self._labels = self._labels.sort_values(
                                    ascending = how == "ascending")
            self.lastSearchType = how
        else:
            self._labels = self._labels.sort_index(
                                    ascending=True)
            self.lastSearchType = None

        self.completeDataChanged()

    def sortByColor(self, e = None, how = "ascending"):
        ""
        sortedColors = self.checkedColors.replace(self.parent().mC.config.getParam("nanColor"),np.nan).sort_values(ascending = how == "ascending")
        self._labels = self._inputLabels.loc[sortedColors.index]
        self.completeDataChanged()


    def showHighlightIndex(self,dataIndex,updateData = False):
        ""
        if dataIndex is None:
            self._labels = self._inputLabels.copy()
        elif isinstance(dataIndex, pd.Series):
            self._labels = self._inputLabels.loc[dataIndex]
        elif isinstance(dataIndex, pd.Int64Index):
            self._labels = self._inputLabels.loc[dataIndex]
        elif isinstance(dataIndex, pd.Float64Index):
            self._labels = self._inputLabels.loc[dataIndex]
        elif dataIndex in self._inputLabels.index:  
            self._labels = self._inputLabels.loc[pd.Series(dataIndex)]
        
        if updateData:
            self.completeDataChanged()
        
        
    def showCheckedLabels(self):
        ""
        if not self.showCheckedOnly:
            checkedIndexes = self.checkedLabels[self.checkedLabels == 1].index
            checkedLabelsIndex = self._labels.index.isin(checkedIndexes)
            self._labels = self._labels[checkedLabelsIndex]
        else:
            self._labels = self._inputLabels
        self.showCheckedOnly = not self.showCheckedOnly
        self.completeDataChanged()

    def completeDataChanged(self):
        ""
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))

    def completeRowChanged(self, rowNumber):
        ""
        self.dataChanged.emit(self.index(rowNumber, 0), self.index(rowNumber, self.columnCount()-1))

    def resetView(self):
        ""
        self._labels = pd.Series()
        self._inputLabels = self._labels.copy()
        self.initColorSeries()
        self.setCheckedSeries() 
        self.completeDataChanged()


class QuickSelectTableView(QTableView):
    def __init__(self, parent=None):
        super(QuickSelectTableView, self).__init__(parent)
        self.setMouseTracking(True)
        self.setShowGrid(False)
        self.verticalHeader().setDefaultSectionSize(12)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.mouseOverItem = None
        self.setAcceptDrops(True)
        self.rightClick = False

        self.setArrowUpAction = QAction("Set arrow up", self, shortcut=Qt.Key_Up, triggered=self.setArrowUp)
        self.addAction(self.setArrowUpAction)

        self.setArrowDownAction = QAction("Set arrow up", self, shortcut=Qt.Key_Down, triggered=self.setArrowDown)
        self.addAction(self.setArrowDownAction)

    def setArrowDown(self):
        ""
        if not self.model().dataAvailable():
            return
        
        if hasattr(self,"mouseOverItem") and isinstance(self.mouseOverItem,int):

            self.mouseOverItem += 1

            index = self.model().index(self.mouseOverItem,0)
            dataIndex = self.model().getDataIndex(self.mouseOverItem)
            
            self.model().setData(index,self.mouseOverItem,Qt.UserRole)
            self.parent().highlightDataInPlotter(dataIndex)

    def setArrowUp(self):
        ""
        if not self.model().dataAvailable():
            return
        
        if hasattr(self,"mouseOverItem") and isinstance(self.mouseOverItem,int):

            self.mouseOverItem -= 1

            index = self.model().index(self.mouseOverItem,0)
            dataIndex = self.model().getDataIndex(self.mouseOverItem)
            self.model().setData(index,self.mouseOverItem,Qt.UserRole)
            
            self.parent().highlightDataInPlotter(dataIndex)

    def resizeColumns(self):
        columnWidths = [(0,200),(1,35),(2,35)]
        for columnId,width in columnWidths:
            self.setColumnWidth(columnId,width)

    def dropEvent(self,e):
        ""
        self.parent().dropEvent(e)

    def enterEvent(self,event):
        ""

    def dragEnterEvent(self,e):
        e.accept()

    def dragMoveEvent(self, e):
        e.accept()

    def mousePressEvent(self,e):
        ""
        if e.buttons() == Qt.RightButton:
            self.rightClick = True
        else:
            self.rightClick = False

    @pyqtSlot()
    def deleteSelection(self):
        ""
        selectionName = self.sender().text()
        self.parent().deleteSelection(selectionName=selectionName)
    
    @pyqtSlot()
    def loadSelection(self):
        ""
        selectionName = self.sender().text()
        self.parent().loadSelection(selectionName=selectionName)
    
    @pyqtSlot()
    def readSelection(self):
        ""
        readType = self.sender().text()
        self.parent().readSelection(readType)
    
    @pyqtSlot()
    def saveSelection(self):
        ""
        self.parent().saveSelection()

    @pyqtSlot()
    def exportSelectionClipboard(self):
        ""
        attachColumns = "all columns" in self.sender().text() 
        self.parent().exportSelectionToClipbard(attachColumns = attachColumns)

    @pyqtSlot()
    def exportSelectionToData(self):
        ""
        self.parent().exportSelectionToData()

    def leaveEvent(self,event=None):
        ""
        if hasattr(self, "mouseOverItem") and self.mouseOverItem is not None:
            prevMouseOver = int(self.mouseOverItem)
            self.mouseOverItem = None
            self.model().completeRowChanged(prevMouseOver)


    def annotateSelection(self):
        "Annotates selection in chart (scatter,hclust)"
        mC = self.parent().mC #get main controller
        if self.parent().getFilterMode() == "raw":
            
            exists, graph = mC.getGraph() #get current graph
            
            if exists:
                if graph.plotType in ["hclust","scatter"]:
                #    print(self.model.checkedLabels.index[self.model.checkedLabels])
                    currentSelection = self.parent().getDataIndexOfCurrentSelection()
                    annotationColumn = self.parent().getQuickSelectColumn()
                    graph.annotateDataByIndex(currentSelection,annotationColumn)
        else:
            mC.sendMessageRequest({"title":"Not available..","message":"Annotations only works for filter mode 'raw'."})
            

    def uncheckSelection(self):
        "Sets check state to unchecked for all items"
        self.model().setCheckedSeries() #resets check state
        self.model().initColorSeries() #resets color
        self.model().setSizeSeries() #resets size
        self.parent().resetGraphItems()
        self.model().completeDataChanged()

    def mouseReleaseEvent(self,e):
        "Handles Mouse Release Events"
        if not self.model().dataAvailable():
                return
        if self.rightClick:
            #handle right click events
            #cast menu
            try:
                savedSelections = self.parent().favSelection.getSavedSelections()
                if len(savedSelections) == 0:
                    subMenus = ["Selection from ..","Export selection"]
                else:
                    subMenus = ["Load","Delete","Selection from ..","Export selection","Add filter legend"]
                menus = createSubMenu(subMenus=subMenus)

                for savedSel in savedSelections:
                    menus["Delete"].addAction(savedSel, self.deleteSelection)
                    menus["Load"].addAction(savedSel, self.loadSelection)

                for readType in ["Clipboard","Text/CSV file"]:
                    menus["Selection from .."].addAction(readType, self.readSelection)

                for legendLocation in legendLocations:
                    menus["Add filter legend"].addAction(legendLocation,lambda loc = legendLocation : self.addLegendForMasking(loc))
                
                menus["Export selection"].addAction("To clipboard", self.exportSelectionClipboard)
                menus["Export selection"].addAction("To clipboard (all columns)", self.exportSelectionClipboard)
                menus["Export selection"].addAction("Add annotation column in data", self.exportSelectionToData)
                
                menus["main"].addAction("Save selection", self.saveSelection)
                menus["main"].addAction("Annotate selection in scatter plot", self.annotateSelection)
                
                menus["main"].addAction("Uncheck all", self.uncheckSelection)
                menus["main"].exec_(self.mapToGlobal(e.pos()))
                    
            except Exception as e:
                print(e)

        else:
            
            tableIndex = self.mouseEventToIndex(e)
            if tableIndex is None:
                return

            if tableIndex.column() == 0:

                self.model().setData(tableIndex,None,Qt.CheckStateRole)

            elif self.model().getCheckStateByTableIndex(tableIndex) and tableIndex.column() == 1:
                color = QColorDialog.getColor()
                if color.isValid():
                    dataIndex = self.model().getDataIndex(tableIndex.row())
                    self.model().setColor(dataIndex,color.name())
            
            elif self.model().getCheckStateByTableIndex(tableIndex) and tableIndex.column() == 2:
                
                currentSize = self.model().getSize(tableIndex)
                sizeInt, okPressed = QInputDialog.getInt(self, "Set Size","Size:",currentSize, 10, 400, 5)
                if okPressed:
                    dataIndex = self.model().getDataIndex(tableIndex.row())
                    self.model().setSize(dataIndex,sizeInt)
        

    def addLegendForMasking(self,loc):
        ""
        print(loc)

    def mouseMoveEvent(self,event):
        ""
        if not self.model().dataAvailable():
            return
        rowAtEvent = self.rowAt(event.pos().y())
        if rowAtEvent == -1:
            return
        self.mouseOverItem = rowAtEvent
        index = self.model().index(self.mouseOverItem,0)
        dataIndex = self.model().getDataIndex(rowAtEvent)
        self.model().setData(index,self.mouseOverItem,Qt.UserRole)
        self.parent().highlightDataInPlotter(dataIndex)

    def mouseEventToIndex(self,event):
        "Converts mouse event on table to tableIndex"
        row = self.rowAt(event.pos().y())
        if row != -1:
            column = self.columnAt(event.pos().x())
            return self.model().index(row,column)
    
    def getCurrentHighlightLabel(self):
        ""
        return self.model().getDataByRow(self.mouseOverItem)
        

        
        

