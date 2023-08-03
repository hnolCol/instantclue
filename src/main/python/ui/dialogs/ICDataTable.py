from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from pandas.core.indexes import multi 

#ui utils
from ..utils import TABLE_ODD_ROW_COLOR, WIDGET_HOVER_COLOR, createTitleLabel, getMessageProps, createLabel
from ..custom.tableviews.ICVSelectableTable import PandaTable, PandaModel, MultiColumnSelectablePandaModel
from ..custom.warnMessage import AskQuestionMessage
from ..custom.Widgets.ICButtonDesgins import ICStandardButton, ResetButton
from ..custom.ICTags import Tag, ICSearchWithTags
#external imports
import pandas as pd 
import numpy as np
import seaborn as sns
from collections import OrderedDict
#to do
#move to dialogs

contextMenuData = OrderedDict([
            ("deleteRows",{"label":"Delete Row(s)","fn":"deleteRows"}),
            ("copyRows",{"label":"Copy Row(s)","fn":"copyRows"}),
            ("copyData",{"label":"Copy Data Frame","fn":"copyDf"})
        ])

headercolors = sns.color_palette("Paired",8,desat=0.75).as_hex()

class PandaTableDialog(QDialog):

    def __init__(self, mainController, df, headerLabel = "Source Data", addToMainDataOption = True, ignoreChanges =  False, filterActive = True, multiSelection = False, modelKwargs = {}, tableKwargs = {}, clippingActive = False, *args, **kwargs):
        super(PandaTableDialog,self).__init__(*args, **kwargs)
        
        self.header = headerLabel
        self.addToMainDataOption = addToMainDataOption
        self.ignoreChanges = ignoreChanges
        self.filterActive = filterActive
        self.multiSelection = multiSelection
        self.df = df
        self.mC = mainController
        self.clippingActive = clippingActive
        self.tagIDColumnIndexMapper = dict()

        self.setHeaderText()
        self.__control(modelKwargs,tableKwargs)
        self.__layout()
        self.__connectEvents()
        self.addData(df)

    def sizeHint(self):
        ""
        return QSize(650,600)

    def __control(self,modelKwargs,tableKwargs):
        ""
        self.headerLabel = createTitleLabel(self.headerLabelText, fontSize=12)
        self.selectionLabel = createLabel("0 row(s) selected")
        self.table = PandaTable(parent= self, mainController = self.mC, rightClickOnHeaderCallBack = self.toggleSearch,**tableKwargs) 
        if not self.multiSelection:
            self.model = PandaModel(parent=self.table,**modelKwargs)#PandaModel()
        else:
            self.model = MultiColumnSelectablePandaModel(parent=self.table,**modelKwargs)
            
        self.table.setModel(self.model)

        if self.filterActive:
            self.filterInfoLabel = createTitleLabel(
                            "Left-click on column header to sort. Right-click on column header to activate search/filter.\nCmd/Ctrl-c allows copying of selected rows to clipboard.", 
                            fontSize=12,
                            colorString="black")
            self.searchWithTags = ICSearchWithTags(onTextChanged = self.handleTextChange,
                                                onEnterEvent=self.handleEnterEvent,
                                                parent=self,
                                                onTagDelete=self.handleTagDelete)
            self.searchWithTags.hideLineEdit()
            self.searchWithTags.hideNumericFilter()
           

        if self.clippingActive:

            self.clippActiveLabel = createTitleLabel("Clipping active!",fontSize=12,colorString=WIDGET_HOVER_COLOR)

           # self.addDataToMain = ICStandardButton(itemName="Save")
        self.closeButton = ResetButton(buttonSize=(20,20))
        self.closeButton.setDefault(False)
        self.closeButton.setAutoDefault(False)

    def __layout(self):
        ""
        self.setLayout(QVBoxLayout())
        hbox = QHBoxLayout()
        hbox.addWidget(self.headerLabel)
        hbox.addStretch(1)
        if hasattr(self,"clippActiveLabel"):
            hbox.addWidget(self.clippActiveLabel)
        hbox.addWidget(self.selectionLabel)
        
       
        hbox.addWidget(self.closeButton)
            
        self.layout().addLayout(hbox)
        if self.filterActive:
            self.layout().addWidget(self.filterInfoLabel)
            self.layout().addWidget(self.searchWithTags)
        self.layout().addWidget(self.table)


    def __connectEvents(self):
        ""
        if hasattr(self,"closeButton"):
            self.closeButton.clicked.connect(self.close)

    def setHeaderText(self):
        ""
        self.headerLabelText = "{} ({} rows x {} columns)".format(self.header,*self.df.shape)
        self.updateHeaderTextOnWidget() 

    def updateHeaderTextOnWidget(self):
        ""
        if hasattr(self,"headerLabel") and hasattr(self.headerLabel,"setText") and hasattr(self,"headerLabelText"):
            self.headerLabel.setText(self.headerLabelText)

    def updateHeaderText(self):
        ""
        self.headerLabelText = "Filtered: ({} rows x {} columns)".format(*self.model.getCurrentShape())
        self.updateHeaderTextOnWidget()

    def handleTagDelete(self,tagID):
        ""
        self.table.model().layoutAboutToBeChanged.emit()
        self.model.removeFilter(tagID)
        self.searchWithTags.removeTag(tagID)
        self.table.model().layoutChanged.emit()
        if tagID in self.tagIDColumnIndexMapper:
            columnIndex = self.tagIDColumnIndexMapper[tagID]
            columnIndexInDict = [columnIdx for columnIdx in self.tagIDColumnIndexMapper.values() if columnIdx == columnIndex]
            if len(columnIndexInDict) == 1:
                self.table.model().removeHighlightBackground(columnIndex)
            del self.tagIDColumnIndexMapper[tagID]

        self.table.model().completeDataChanged()
        if self.table.model().filters.columns.size == 0:
            self.setHeaderText()
        else:
            self.updateHeaderText()

    def handleTextChange(self,searchText):
        ""
        self.currentSearchText = searchText

    def handleEnterEvent(self,event = None, exactMatch = False, filterType="categorical", minValue = None, maxValue = None):
        ""
    
        backgroundColor = self.model.getHighlightBackgroundcolor(self.idxClicked)
        columnName = self.model.getColumnNameByColumnIndex(self.idxClicked)
        if backgroundColor is None:
            nHighlightedColors = self.model.getNumberOfHighlightedBackgrounds()
            backgroundColor = headercolors[nHighlightedColors % len(headercolors)]
            self.model.setHighlightBackground(self.idxClicked,backgroundColor)
        if filterType == "categorical":
            tagID = self.searchWithTags.addTag(self.currentSearchText,backgroundColor)
            self.table.model().layoutAboutToBeChanged.emit()
            self.table.model().setFilterByColumnIndex(self.idxClicked,self.currentSearchText,tagID,exactMatch)
            self.table.model().layoutChanged.emit()
            self.table.model().completeDataChanged()
            self.searchWithTags.hideLineEdit()
        else:
                   
            tagID = self.searchWithTags.addTag("{}:{}-{}".format(columnName,minValue,maxValue),backgroundColor)
            self.table.model().layoutAboutToBeChanged.emit()
            self.table.model().setNumericFilterByColumnIndex(self.idxClicked,minValue,maxValue,tagID)
            self.table.model().layoutChanged.emit()
            self.table.model().completeDataChanged()
            self.searchWithTags.hideNumericFilter()
        self.updateHeaderText()
        self.tagIDColumnIndexMapper[tagID] = int(self.idxClicked)


    def toggleSearch(self,idxClicked):
        ""
        if self.filterActive:
            self.idxClicked = idxClicked
            columnName = self.model.getColumnNameByColumnIndex(idxClicked)
            dataType = self.model.getDataTypeByColumnIndex(idxClicked)
            
            if columnName is not None and dataType not in [np.float64,np.int64]:
                self.searchWithTags.showLineEdit()
                self.searchWithTags.hideNumericFilter()
                self.searchWithTags.resetLineEditText()
                self.searchWithTags.setFocusToLineEdit()
                self.searchWithTags.setPlaceHolderText("Search/Filter in "+columnName)
                
            else:
                self.searchWithTags.hideLineEdit()
                self.searchWithTags.showNumericFilter(columnName)
                self.searchWithTags.setFocusToMin()
        
    def sendMessage(self, messageProps):
        ""
        if hasattr(self.parent(),"mC"):
            self.parent().mC.sendMessageRequest(messageProps) 

    def sendToThread(self,funcProps):
        ""
        if hasattr(self.parent(),"mC"):
            self.parent().mC.sendRequestToThread(funcProps)

    def setSelectedRowsLabel(self, nRows = 0):
        ""
        if isinstance(nRows,int):
            self.selectionLabel.setText("{} row(s) selected".format(nRows))

    def addData(self,X = pd.DataFrame()):
        ""
        if isinstance(X,pd.DataFrame):
            self.table.model().layoutAboutToBeChanged.emit()
            self.table.model().updateDataFrame(X)
            self.table.model().layoutChanged.emit()

    def closeEvent(self,e=None):
        ""
        
        #set hover elements invisible
        exists,graph = self.mC.getGraph()
        if exists:
            graph.setHoverObjectsInvisible()
            graph.updateFigure.emit()

        if self.ignoreChanges:
            e.accept()
        elif self.df.equals(self.table.model().df):
            e.accept()
        else:
            quest = AskQuestionMessage(title = "Question", infoText = "Data have changed. Update data?")
            quest.exec()
            if quest.state:
                questForCopy = AskQuestionMessage(title = "Question", 
                        infoText = "Would you like to update the current data in place?\n\n(no - creates a new data frame with the changes made (sorting and filtering included))?")
                questForCopy.exec()
                if questForCopy.state:
                    funcProps = dict() 
                    funcProps["key"] = "data::updateData"
                    funcProps["kwargs"] = {"dataID":self.mC.getDataID(),"data":self.model.df}
                    self.sendToThread(funcProps)
                else:
                    funcProps = dict() 
                    fileName = self.mC.data.getFileNameByID(self.mC.getDataID())
                    funcProps["key"] = "data::addDataFrame"
                    funcProps["kwargs"] = {"fileName":"updated({})".format(fileName),"dataFrame":self.model.df}
                    self.sendToThread(funcProps)
            self.close()
    


class ICLabelDataTableDialog(PandaTableDialog):
    ""
    def __init__(self,modelKwargs = {},tableKwargs = {},*args,**kwargs):
        ""
        super(ICLabelDataTableDialog,self).__init__(*args,**kwargs, ignoreChanges=True ,addToMainDataOption=False, filterActive=False, multiSelection=True, modelKwargs = modelKwargs,tableKwargs=tableKwargs)

    



    
